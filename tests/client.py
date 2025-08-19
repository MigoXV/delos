# client.py
import argparse
import time
import numpy as np
import grpc

from delos.proto import env_pb2, env_pb2_grpc


def main(addr: str, seed: int, steps: int):
    # 连接到服务端
    channel = grpc.insecure_channel(addr)
    stub = env_pb2_grpc.EnvStub(channel)

    # 读环境规格，确认形状与边界
    spec = stub.Spec(env_pb2.google_dot_protobuf_dot_empty__pb2.Empty())
    print(f"Connected to {spec.env_id}")
    obs_shape = tuple(spec.observation.shape)
    act_shape = tuple(spec.action.shape)
    act_low = np.array(spec.action.low, dtype=np.float32).reshape(act_shape)
    act_high = np.array(spec.action.high, dtype=np.float32).reshape(act_shape)

    # 复位（可设种子）
    reset_resp = stub.Reset(env_pb2.ResetRequest(seed=seed))
    obs = np.array(reset_resp.observation, dtype=np.float32).reshape(obs_shape)

    ep_return, t = 0.0, 0
    while t < steps:
        # 随机动作（示范）：真实训练时改为 policy(obs) 推断
        action = np.random.uniform(low=act_low, high=act_high, size=act_shape).astype(
            np.float32
        )

        # 发送一步
        step_resp = stub.Step(env_pb2.StepRequest(action=action.ravel().tolist()))
        obs = np.array(step_resp.observation, dtype=np.float32).reshape(obs_shape)
        ep_return += step_resp.reward
        t += 1

        # 处理终止/截断并重置
        if step_resp.terminated or step_resp.truncated:
            print(f"Episode done at t={t}, return={ep_return:.2f}")
            ep_return, t = 0.0, 0
            reset_resp = stub.Reset(env_pb2.ResetRequest(seed=-1))
            obs = np.array(reset_resp.observation, dtype=np.float32).reshape(obs_shape)

        # 可选：慢放观察
        # time.sleep(0.01)

    # 结束前关闭
    stub.Close(env_pb2.google_dot_protobuf_dot_empty__pb2.Empty())
    channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--addr", type=str, default="127.0.0.1:50051", help="server 地址"
    )
    parser.add_argument("--seed", type=int, default=42, help="复位种子，<0 表示不设")
    parser.add_argument("--steps", type=int, default=1000, help="最多步数")
    args = parser.parse_args()
    main(args.addr, args.seed, args.steps)
