import argparse

from delos.proto.server import serve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=50051, help="监听端口")
    parser.add_argument(
        "--env-id", type=str, default="Pendulum-v1", help="Gymnasium 环境 ID"
    )
    args = parser.parse_args()
    serve(args.host, args.port, args.env_id)