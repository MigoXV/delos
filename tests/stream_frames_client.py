#!/usr/bin/env python3
"""测试流式视频帧客户端"""

import grpc
from delos.proto import env_pb2, env_pb2_grpc
from google.protobuf import empty_pb2
import time
import os


def save_frame(frame_data: bytes, frame_number: int, output_dir: str = "frames"):
    """保存帧到文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"{output_dir}/frame_{frame_number:04d}.png"
    with open(filename, "wb") as f:
        f.write(frame_data)
    print(f"保存帧: {filename}")


def main():
    # 连接到服务器
    channel = grpc.insecure_channel("localhost:50051")
    stub = env_pb2_grpc.EnvStub(channel)
    
    try:
        # 首先获取环境规格
        print("获取环境规格...")
        spec = stub.Spec(empty_pb2.Empty())
        print(f"环境ID: {spec.env_id}")
        print(f"观测空间: {spec.observation}")
        print(f"动作空间: {spec.action}")
        
        # 重置环境
        print("\n重置环境...")
        reset_response = stub.Reset(env_pb2.ResetRequest(seed=42))
        print(f"初始观测: {list(reset_response.observation)[:5]}...")  # 只显示前5个值
        
        # 创建流式帧请求
        frame_request = env_pb2.FrameRequest(
            width=320,   # 可选：指定帧宽度
            height=240,  # 可选：指定帧高度
            mode="rgb_array"  # 渲染模式
        )
        
        print(f"\n开始流式获取视频帧...")
        frame_count = 0
        max_frames = 30  # 最多获取30帧
        
        # 开始流式获取帧
        for frame_response in stub.StreamFrames(frame_request):
            if frame_response.has_frame:
                print(f"接收到帧 {frame_count + 1}: "
                      f"{frame_response.width}x{frame_response.height}, "
                      f"格式: {frame_response.format}, "
                      f"大小: {len(frame_response.image_data)} 字节, "
                      f"时间戳: {frame_response.timestamp}")
                
                # 保存前几帧作为示例
                if frame_count < 5:
                    save_frame(frame_response.image_data, frame_count)
                
                frame_count += 1
                
                # 在流式传输过程中执行一些环境步骤
                if frame_count % 10 == 0:
                    # 每10帧执行一个随机动作
                    import random
                    if spec.action.type == "Box":
                        action_dim = spec.action.shape[0]
                        action = [random.uniform(-1, 1) for _ in range(action_dim)]
                    elif spec.action.type == "Discrete":
                        # 对于离散动作空间，发送单个整数值作为浮点数
                        action = [float(random.randint(0, 1))]  # CartPole 有2个动作：0或1
                    else:
                        action = [0.0]
                    
                    step_response = stub.Step(env_pb2.StepRequest(action=action))
                    print(f"  执行动作后 - 奖励: {step_response.reward:.3f}, "
                          f"结束: {step_response.terminated}, "
                          f"截断: {step_response.truncated}")
            else:
                print(f"帧 {frame_count + 1}: 无有效帧数据")
                frame_count += 1
            
            # 限制帧数
            if frame_count >= max_frames:
                print(f"达到最大帧数限制 {max_frames}，停止流式传输")
                break
                
    except grpc.RpcError as e:
        print(f"gRPC 错误: {e.code()}: {e.details()}")
    except KeyboardInterrupt:
        print("\n用户中断，停止流式传输")
    except Exception as e:
        print(f"客户端错误: {e}")
    finally:
        # 关闭环境
        try:
            stub.Close(empty_pb2.Empty())
            print("环境已关闭")
        except:
            pass
        channel.close()


if __name__ == "__main__":
    main()
