import asyncio
import math
import random
import grpc
import pygame

from delos.protos.cartpole import env_pb2 as pb
from delos.protos.cartpole import env_pb2_grpc as pb_grpc


# 画面与物理到像素映射参数
SCREEN_W, SCREEN_H = 900, 500           # 画布大小
X_THRESHOLD = 2.4                        # CartPole 合法位移阈值（典型值）
SIDE_MARGIN_WORLD = 0.6                  # 视觉留白（世界坐标）
CART_W, CART_H = 70, 35                  # 小车尺寸（像素）
POLE_LEN_PX = 140                        # 杆长（像素，可调）
AXLE_RADIUS = 6                          # 转轴小圆
FLOOR_Y = int(SCREEN_H * 0.75)           # 地面高度
BG_COLOR = (248, 249, 250)
LINE_COLOR = (210, 215, 220)
CART_COLOR = (80, 110, 160)
POLE_COLOR = (30, 30, 30)
TEXT_COLOR = (20, 20, 20)
DONE_OVERLAY = (255, 70, 70)


class PygameRenderer:
    def __init__(self) -> None:
        # 初始化 pygame
        pygame.init()
        pygame.display.set_caption("CartPole-v1 Viewer (gRPC streaming)")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        # 世界坐标到像素的横向比例尺
        world_half = X_THRESHOLD + SIDE_MARGIN_WORLD
        self.x_scale = SCREEN_W / (2 * world_half)

        # 控制模式与状态
        self.manual_mode = False          # False=自动（基于角度），True=手动
        self.auto_reset = True            # 回合结束是否自动 reset
        self.manual_action = None         # 在手动模式下，记录上一次键盘动作

    def _world_to_px(self, x_world: float) -> int:
        # 世界坐标 x (米) -> 屏幕像素
        # 屏幕中心对应 x=0
        return int(SCREEN_W / 2 + x_world * self.x_scale)

    def handle_events(self) -> tuple[bool, bool, int | None]:
        # 处理窗口/键盘事件；返回 (quit, want_reset, manual_action)
        quit_now = False
        want_reset = False
        manual_action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_now = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    quit_now = True
                elif event.key == pygame.K_r:
                    want_reset = True
                elif event.key == pygame.K_a:
                    # 切换自动/手动控制
                    self.manual_mode = not self.manual_mode
                elif event.key == pygame.K_m:
                    # 同样切换（备选）
                    self.manual_mode = not self.manual_mode
                elif event.key == pygame.K_LEFT:
                    self.manual_mode = True
                    manual_action = 0
                    self.manual_action = 0
                elif event.key == pygame.K_RIGHT:
                    self.manual_mode = True
                    manual_action = 1
                    self.manual_action = 1
        return quit_now, want_reset, manual_action

    def render(self, obs: list[float], step_idx: int | None, reward: float | None, done: bool) -> None:
        # obs = [x, x_dot, theta, theta_dot]
        x = float(obs[0])
        theta = float(obs[2])

        # 背景与地面
        self.screen.fill(BG_COLOR)
        pygame.draw.line(self.screen, LINE_COLOR, (0, FLOOR_Y), (SCREEN_W, FLOOR_Y), 2)

        # 轨道（±X_THRESHOLD）
        x_left = self._world_to_px(-X_THRESHOLD)
        x_right = self._world_to_px(+X_THRESHOLD)
        pygame.draw.line(self.screen, (180, 190, 200), (x_left, FLOOR_Y), (x_right, FLOOR_Y), 4)

        # 小车位置
        cart_x_center = self._world_to_px(x)
        cart_rect = pygame.Rect(0, 0, CART_W, CART_H)
        cart_rect.midbottom = (cart_x_center, FLOOR_Y)
        pygame.draw.rect(self.screen, CART_COLOR, cart_rect, border_radius=8)

        # 杆：pivot 在小车顶中点；屏幕坐标 y 向下为正，所以用 -theta 作视觉修正
        pivot = (cart_x_center, cart_rect.top)
        angle = -theta
        end_x = pivot[0] + int(POLE_LEN_PX * math.sin(angle))
        end_y = pivot[1] - int(POLE_LEN_PX * math.cos(angle))
        pygame.draw.line(self.screen, POLE_COLOR, pivot, (end_x, end_y), 6)
        pygame.draw.circle(self.screen, (50, 50, 50), pivot, AXLE_RADIUS)

        # HUD
        hud_lines = [
            f"mode: {'MANUAL' if self.manual_mode else 'AUTO'}   auto-reset: {'ON' if self.auto_reset else 'OFF'}",
            f"x={x:+.3f}  theta={theta:+.3f} rad   {'DONE' if done else ''}",
        ]
        if step_idx is not None and reward is not None:
            hud_lines.append(f"step={step_idx}  reward={reward:.1f}")
        for i, text in enumerate(hud_lines):
            surf = self.font.render(text, True, TEXT_COLOR)
            self.screen.blit(surf, (12, 12 + i * 22))

        # 回合结束叠加提示
        if done:
            overlay = self.font.render("EPISODE DONE", True, DONE_OVERLAY)
            self.screen.blit(overlay, (12, 12 + len(hud_lines) * 22))

        pygame.display.flip()
        self.clock.tick(60)  # 限制刷新率；gRPC 步进节奏由对端/本端策略决定


async def client_loop():
    # 用队列把“要发给服务端的请求”从消费端（渲染/策略）异步推送给生产端（gRPC 发送）
    action_q: asyncio.Queue[pb.EnvRequest] = asyncio.Queue()
    stop = asyncio.Event()

    # request_id 简单递增
    req_counter = 0
    def next_req_id(prefix: str) -> str:
        nonlocal req_counter
        req_counter += 1
        return f"{prefix}-{req_counter}"

    # 组装几种请求
    def make_reset(force: bool = True, seed: int | None = None) -> pb.EnvRequest:
        return pb.EnvRequest(
            request_id=next_req_id("reset"),
            episode_id="",
            env_id="CartPole-v1",
            reset=pb.Reset(force=force, seed=seed if seed is not None else 42),
        )

    def make_step(act: int) -> pb.EnvRequest:
        return pb.EnvRequest(
            request_id=next_req_id("step"),
            episode_id="",
            step=pb.Step(action=int(act)),
        )

    def make_close(reason: str) -> pb.EnvRequest:
        return pb.EnvRequest(
            request_id=next_req_id("close"),
            episode_id="",
            close=pb.Close(reason=reason),
        )

    # 初始化渲染器
    viewer = PygameRenderer()

    # 先压入一次 reset
    await action_q.put(make_reset(force=True, seed=42))

    # gRPC 通道
    async with grpc.aio.insecure_channel("localhost:56111") as channel:
        stub = pb_grpc.CartPoleServiceStub(channel)

        # 发送端：从队列拿请求并 yield 给 gRPC
        async def request_gen():
            while not stop.is_set():
                req = await action_q.get()
                yield req
                # 适当小憩，避免疯狂灌流；节奏以“收到响应后再投下一步”为主
                await asyncio.sleep(0)

        # 接收端：处理响应、渲染、决定下一步动作
        try:
            async for resp in stub.EnvStream(request_gen()):
                # 统一处理窗口事件（随时可退出、重置、切换模式）
                quit_now, want_reset, manual_action = viewer.handle_events()
                if quit_now:
                    await action_q.put(make_close("user-quit"))
                    stop.set()

                if want_reset:
                    await action_q.put(make_reset(force=True))

                # 根据不同响应分支进行渲染 & 决策
                if resp.HasField("error"):
                    print(f"[ERROR] code={resp.error.code} msg={resp.error.message}")
                    await action_q.put(make_close("error"))
                    stop.set()
                    break

                elif resp.HasField("reset_result"):
                    obs = list(resp.reset_result.observation)
                    viewer.render(obs, step_idx=None, reward=None, done=False)

                    # Reset 后投第一步动作（自动或手动）
                    if viewer.manual_mode and viewer.manual_action is not None:
                        await action_q.put(make_step(viewer.manual_action))
                    else:
                        # 简单策略：θ>0 往右推（1），θ<0 往左推（0）
                        theta = float(obs[2])
                        act = 1 if theta > 0 else 0
                        await action_q.put(make_step(act))

                elif resp.HasField("step_result"):
                    obs = list(resp.step_result.observation)
                    rew = float(resp.step_result.reward)
                    done = bool(resp.step_result.terminated or resp.step_result.truncated)
                    viewer.render(obs, step_idx=int(resp.step_index), reward=rew, done=done)

                    if done:
                        # 回合结束：根据设置自动重置或等待按键 R
                        if viewer.auto_reset:
                            await asyncio.sleep(0.2)
                            await action_q.put(make_reset(force=True))
                        else:
                            # 不自动重置时，不再下发 step，等待用户 R
                            pass
                    else:
                        # 正常推进：每收一帧，恰好投递一个下一步
                        if viewer.manual_mode and (manual_action is not None or viewer.manual_action is not None):
                            act = manual_action if manual_action is not None else viewer.manual_action
                            await action_q.put(make_step(int(act)))
                        else:
                            theta = float(obs[2])
                            act = 1 if theta > 0 else 0
                            await action_q.put(make_step(act))

                elif resp.HasField("closed"):
                    print(f"[CLOSED] {resp.closed.message}")
                    stop.set()
                    break

        finally:
            # 退出清理
            try:
                if not stop.is_set():
                    await action_q.put(make_close("shutdown"))
            except Exception:
                pass
            pygame.quit()


if __name__ == "__main__":
    # 直接运行：显示窗口，自动控制，按 A 可切换手动，←/→ 施加动作
    asyncio.run(client_loop())
