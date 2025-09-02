import asyncio
import math
import grpc
import pygame

from delos.protos.cartpole import env_pb2 as pb
from delos.protos.cartpole import env_pb2_grpc as pb_grpc


# ====== 画面与物理到像素映射参数 ======
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
    """仅手动控制：每一步都等待键盘输入 (A=0, D=1)。
    Q/ESC 退出，R 重置。
    """

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("CartPole-v1 Viewer (gRPC streaming, MANUAL A/D)")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        world_half = X_THRESHOLD + SIDE_MARGIN_WORLD
        self.x_scale = SCREEN_W / (2 * world_half)

    def _world_to_px(self, x_world: float) -> int:
        # 世界坐标 x (米) -> 屏幕像素（屏幕中心对应 x=0）
        return int(SCREEN_W / 2 + x_world * self.x_scale)

    def _draw_frame(self, obs: list[float], step_idx: int | None, reward: float | None, done: bool, hint: str | None = None) -> None:
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

        # 小车
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
            "mode: MANUAL (A=0, D=1)   R=reset   Q/ESC=quit",
            f"x={x:+.3f}  theta={theta:+.3f} rad   {'DONE' if done else ''}",
        ]
        if step_idx is not None and reward is not None:
            hud_lines.append(f"step={step_idx}  reward={reward:.1f}")
        if hint:
            hud_lines.append(hint)
        for i, text in enumerate(hud_lines):
            surf = self.font.render(text, True, TEXT_COLOR)
            self.screen.blit(surf, (12, 12 + i * 22))

        if done:
            overlay = self.font.render("EPISODE DONE", True, DONE_OVERLAY)
            self.screen.blit(overlay, (12, 12 + len(hud_lines) * 22))

        pygame.display.flip()

    async def wait_for_action_or_cmd(self, obs: list[float], step_idx: int | None, reward: float | None, done: bool) -> tuple[bool, bool, int | None]:
        """循环处理事件，直到：
        - 用户按下 A/D -> 返回 (quit=False, want_reset=False, action)
        - 用户按下 R   -> 返回 (quit=False, want_reset=True,  None)
        - 用户按下 Q/ESC 或 关闭窗口 -> 返回 (quit=True,  want_reset=False, None)
        期间保持以 ~60FPS 重绘当前帧。
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True, False, None
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        return True, False, None
                    elif event.key == pygame.K_r:
                        return False, True, None
                    elif event.key == pygame.K_a:
                        return False, False, 0
                    elif event.key == pygame.K_d:
                        return False, False, 1
            # 每次循环都把提示画上
            hint = "Press A (0) or D (1) to step" if not done else "Episode done. Press R to reset."
            self._draw_frame(obs, step_idx, reward, done, hint=hint)
            self.clock.tick(60)
            await asyncio.sleep(0)  # 让出事件循环


async def client_loop():
    """手动逐步：每接收一帧，就等待用户按 A/D 再发送下一步。"""
    action_q: asyncio.Queue[pb.EnvRequest] = asyncio.Queue()
    stop = asyncio.Event()

    # request_id 简单递增
    req_counter = 0
    def next_req_id(prefix: str) -> str:
        nonlocal req_counter
        req_counter += 1
        return f"{prefix}-{req_counter}"

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

    viewer = PygameRenderer()

    # 先 reset 一次
    await action_q.put(make_reset(force=True, seed=42))

    async with grpc.aio.insecure_channel("localhost:56111") as channel:
        stub = pb_grpc.CartPoleServiceStub(channel)

        async def request_gen():
            while not stop.is_set():
                req = await action_q.get()
                yield req
                await asyncio.sleep(0)

        try:
            async for resp in stub.EnvStream(request_gen()):
                # 根据响应分支进行渲染与下一步输入
                if resp.HasField("error"):
                    print(f"[ERROR] code={resp.error.code} msg={resp.error.message}")
                    await action_q.put(make_close("error"))
                    stop.set()
                    break

                elif resp.HasField("reset_result"):
                    obs = list(resp.reset_result.observation)
                    # reset 后等待用户按键：A/D 发送一步；R 继续 reset；Q 退出
                    quit_now, want_reset, action = await viewer.wait_for_action_or_cmd(obs, None, None, False)
                    if quit_now:
                        await action_q.put(make_close("user-quit"))
                        stop.set()
                    elif want_reset:
                        await action_q.put(make_reset(force=True))
                    elif action is not None:
                        await action_q.put(make_step(action))

                elif resp.HasField("step_result"):
                    obs = list(resp.step_result.observation)
                    rew = float(resp.step_result.reward)
                    done = bool(resp.step_result.terminated or resp.step_result.truncated)

                    # 等待本步的用户指令
                    quit_now, want_reset, action = await viewer.wait_for_action_or_cmd(obs, int(resp.step_index), rew, done)
                    if quit_now:
                        await action_q.put(make_close("user-quit"))
                        stop.set()
                        break
                    if want_reset:
                        await action_q.put(make_reset(force=True))
                        continue
                    if done:
                        # 回合已结束，必须 R 才能继续
                        # 如果用户误按 A/D，就忽略，上面的 wait_for_action_or_cmd 在 done 时只提供 R
                        continue
                    if action is not None:
                        await action_q.put(make_step(action))

                elif resp.HasField("closed"):
                    print(f"[CLOSED] {resp.closed.message}")
                    stop.set()
                    break
        finally:
            try:
                if not stop.is_set():
                    await action_q.put(make_close("shutdown"))
            except Exception:
                pass
            pygame.quit()


if __name__ == "__main__":
    # 运行：每一步都按 A (action=0) 或 D (action=1)，R 重置，Q/ESC 退出
    asyncio.run(client_loop())
