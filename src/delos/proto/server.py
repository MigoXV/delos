from concurrent import futures

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


def serve(host: str, port: int, env_id: str):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1)
    )  # 单线程，保证环境安全
    env_pb2_grpc.add_EnvServicer_to_server(EnvServicer(env_id), server)
    server.add_insecure_port(f"{host}:{port}")  # 本机开发用明文端口即可
    server.start()
    print(f"[gRPC Env] Serving {env_id} on {host}:{port}")
    server.wait_for_termination()
