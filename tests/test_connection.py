#!/usr/bin/env python3
"""简单的 gRPC 连接测试"""

import grpc
from delos.proto import env_pb2, env_pb2_grpc
from google.protobuf import empty_pb2


def test_connection():
    """测试 gRPC 连接"""
    try:
        # 创建连接
        channel = grpc.insecure_channel("localhost:50051")
        stub = env_pb2_grpc.EnvStub(channel)
        
        # 测试连接
        print("尝试连接到服务器...")
        
        # 设置超时时间
        spec = stub.Spec(empty_pb2.Empty(), timeout=5.0)
        print(f"连接成功！环境ID: {spec.env_id}")
        
        # 测试重置
        reset_response = stub.Reset(env_pb2.ResetRequest(seed=42))
        print(f"重置成功，观测维度: {len(reset_response.observation)}")
        
        # 关闭连接
        channel.close()
        print("连接测试完成")
        
    except grpc.RpcError as e:
        print(f"gRPC 错误: {e.code()}: {e.details()}")
    except Exception as e:
        print(f"其他错误: {e}")


if __name__ == "__main__":
    test_connection()
