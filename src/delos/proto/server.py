from concurrent import futures
import time

import grpc
from google.protobuf import empty_pb2

from delos.proto import env_pb2, env_pb2_grpc
from delos.proto.grpc_player import GrpcPlayer

def _dict_to_space_spec(info: dict) -> env_pb2.SpaceSpec:
    """将仿真返回的空间 dict 转为 SpaceSpec。"""
    spec = env_pb2.SpaceSpec()
    spec.type = info.get("type", "")
    for v in info.get("shape", []) or []:
        spec.shape.append(int(v))
    # 仅 Box 有 low/high
    for v in info.get("low", []) or []:
        spec.low.append(float(v))
    for v in info.get("high", []) or []:
        spec.high.append(float(v))
    return spec


class EnvServicer(env_pb2_grpc.EnvServicer):
    """gRPC 服务实现：仅负责将消息与 Player 的纯 Python 数据互转。"""

    def __init__(self, env_id: str):
        self.player = GrpcPlayer(env_id)

    def Spec(self, request, context):
        # 返回空间元数据：从 player 获取并构建 protobuf
        spec_dict = self.player.get_spec()
        obs_spec = _dict_to_space_spec(spec_dict["observation"])
        act_spec = _dict_to_space_spec(spec_dict["action"])
        return env_pb2.SpecResponse(
            env_id=spec_dict["env_id"], observation=obs_spec, action=act_spec
        )

    def Reset(self, request, context):
        observation = self.player.reset(seed=request.seed)
        return env_pb2.ResetResponse(observation=observation)

    def Step(self, request, context):
        result = self.player.step(list(request.action))
        return env_pb2.StepResponse(
            observation=result["observation"],
            reward=result["reward"],
            terminated=result["terminated"],
            truncated=result["truncated"],
        )

    def Close(self, request, context):
        self.player.close()
        return empty_pb2.Empty()

    def StreamFrames(self, request, context):
        """流式传输环境视频帧"""
        # 提取请求参数
        width = request.width if request.width > 0 else None
        height = request.height if request.height > 0 else None
        mode = request.mode if request.mode else "rgb_array"
        
        try:
            frame_count = 0
            max_frames = 1000  # 限制最大帧数，防止无限流式传输
            
            while frame_count < max_frames:
                # 检查客户端是否仍然连接
                if context.is_active():
                    # 渲染当前帧
                    frame_data = self.player.render_frame(width=width, height=height, mode=mode)
                    
                    # 构建响应
                    response = env_pb2.FrameResponse(
                        image_data=frame_data["image_data"],
                        width=frame_data["width"],
                        height=frame_data["height"],
                        format=frame_data["format"],
                        timestamp=frame_data["timestamp"],
                        has_frame=frame_data["has_frame"]
                    )
                    
                    yield response
                    frame_count += 1
                    
                    # 控制帧率，大约 30 FPS
                    time.sleep(1.0 / 30.0)
                else:
                    print("客户端断开连接")
                    break
                
        except Exception as e:
            print(f"流式传输视频帧时出错: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"流式传输错误: {str(e)}")
        finally:
            print("视频帧流式传输结束")


def serve(host: str, port: int, env_id: str):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1)
    )  # 单线程，保证环境安全
    env_pb2_grpc.add_EnvServicer_to_server(EnvServicer(env_id), server)
    server.add_insecure_port(f"{host}:{port}")  # 本机开发用明文端口即可
    server.start()
    print(f"[gRPC Env] Serving {env_id} on {host}:{port}")
    server.wait_for_termination()
