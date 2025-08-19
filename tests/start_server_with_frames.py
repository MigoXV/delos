#!/usr/bin/env python3
"""启动支持视频流的环境服务器"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from delos.proto.server import serve

if __name__ == "__main__":
    # 使用支持渲染的环境，比如 CartPole
    env_id = "CartPole-v1"
    host = "localhost"
    port = 50051
    
    print(f"启动环境服务器: {env_id}")
    print(f"服务地址: {host}:{port}")
    print("按 Ctrl+C 停止服务器")
    
    try:
        serve(host, port, env_id)
    except KeyboardInterrupt:
        print("\n服务器已停止")
