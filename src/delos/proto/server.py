# server.py
import argparse                         # 解析命令行参数
from concurrent import futures          # gRPC 线程池
import threading                        # 可加锁保护（若多环境）
import numpy as np                      # 数值处理
import grpc                             # gRPC 核心
import gymnasium as gym                 # 环境
from google.protobuf import empty_pb2   # Empty 消息

from delos.proto import env_pb2                          # protoc 生成
from delos.proto import env_pb2_grpc                     # protoc 生成

def _flatten(x):
    """将任意形状的数组展平成 1D float32 列表。"""
    return np.asarray(x, dtype=np.float32).ravel().tolist()

def _space_to_spec(space):
    """将 Gymnasium 空间描述转为 SpaceSpec。仅示范 Box。"""
    spec = env_pb2.SpaceSpec()
    if space.__class__.__name__ == "Box":
        spec.type = "Box"
        spec.shape.extend(list(space.shape))
        # 上下界：无穷大/小用 float('inf') 表示；转为 float32 更紧凑
        spec.low.extend(np.asarray(space.low, dtype=np.float32).ravel().tolist())
        spec.high.extend(np.asarray(space.high, dtype=np.float32).ravel().tolist())
    else:
        spec.type = space.__class__.__name__
        if hasattr(space, "shape") and space.shape is not None:
            spec.shape.extend(list(space.shape))
    return spec

class EnvServicer(env_pb2_grpc.EnvServicer):
    """gRPC 服务实现：封装单个 Gymnasium 环境。"""
    def __init__(self, env_id: str):
        # 创建环境；渲染若要取图像可设 render_mode="rgb_array"
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.lock = threading.Lock()  # 若未来并发访问，这里可保护临界区

    def Spec(self, request, context):
        # 返回空间元数据：便于客户端自检
        obs_spec = _space_to_spec(self.env.observation_space)
        act_spec = _space_to_spec(self.env.action_space)
        return env_pb2.SpecResponse(env_id=self.env_id,
                                    observation=obs_spec,
                                    action=act_spec)

    def Reset(self, request, context):
        # 可选种子：<0 或未设 -> 不设定
        seed = request.seed if request.seed >= 0 else None
        with self.lock:
            obs, _info = self.env.reset(seed=seed)
        return env_pb2.ResetResponse(observation=_flatten(obs))

    def Step(self, request, context):
        # 将动作列表还原为 numpy 并裁剪到动作空间范围（保险）
        with self.lock:
            act_space = self.env.action_space
            # Pendulum 动作是 shape=(1,) 的 Box
            if hasattr(act_space, "shape") and act_space.shape is not None:
                action = np.array(request.action, dtype=np.float32)
                action = action.reshape(act_space.shape)
                # 若有上下界，进行裁剪，避免数值越界
                if hasattr(act_space, "low"):
                    low = np.asarray(act_space.low, dtype=np.float32)
                    high = np.asarray(act_space.high, dtype=np.float32)
                    action = np.clip(action, low, high)
            else:
                # 退化处理：直接取第一个标量
                action = float(request.action[0]) if request.action else 0.0

            obs, reward, terminated, truncated, _info = self.env.step(action)

        return env_pb2.StepResponse(
            observation=_flatten(obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

    def Close(self, request, context):
        with self.lock:
            self.env.close()
        return empty_pb2.Empty()

def serve(host: str, port: int, env_id: str):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))  # 单线程，保证环境安全
    env_pb2_grpc.add_EnvServicer_to_server(EnvServicer(env_id), server)
    server.add_insecure_port(f"{host}:{port}")  # 本机开发用明文端口即可
    server.start()
    print(f"[gRPC Env] Serving {env_id} on {host}:{port}")
    server.wait_for_termination()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=50051, help="监听端口")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1", help="Gymnasium 环境 ID")
    args = parser.parse_args()
    serve(args.host, args.port, args.env_id)
