from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Optional, Tuple

import grpc
import gymnasium as gym

from delos.protos.cartpole import env_pb2 as pb
from delos.protos.cartpole import env_pb2_grpc as pb_grpc

# CartPole-v1 streaming gRPC server (Python, asyncio)
# - No rendering
# - Bidirectional streaming via EnvStream
# - Commands carried in oneof: Reset | Step | Close
# - Imports expect generated code under delos.protos.cartpole.*
#
# Start: python server.py


class Session:
    """Per-stream episode/session state."""

    def __init__(self) -> None:
        self.env: Optional[object] = None  # gym.Env, but avoid hard dependency in types
        self.episode_id: str = ""
        self.step_index: int = -1  # becomes 0 right after reset
        self.last_observation: Optional[list[float]] = None

    def close_env(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
            self.step_index = -1
            self.last_observation = None


class CartPoleService(pb_grpc.CartPoleServiceServicer):
    async def EnvStream(
        self,
        request_iterator: AsyncIterator[pb.EnvRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.EnvResponse]:
        session = Session()
        async for req in request_iterator:
            now_ms = int(time.time() * 1000)
            rid = req.request_id
            eid = req.episode_id

            # Handle Reset
            if req.HasField("reset"):
                # Close existing env if force is set
                if session.env is not None and req.reset.force:
                    session.close_env()

                # (Re)create env lazily; no render_mode
                if session.env is None:
                    session.env = gym.make(req.env_id or "CartPole-v1")

                # Seed if requested (gymnasium supports seed in reset)
                seed_arg = req.reset.seed if req.reset.seed != 0 else None
                obs, info = session.env.reset(seed=seed_arg)
                session.episode_id = eid or session.episode_id or _gen_episode_id()
                session.step_index = 0
                session.last_observation = _to_list(obs)

                resp = pb.EnvResponse(
                    request_id=rid,
                    episode_id=session.episode_id,
                    step_index=session.step_index,
                    server_time_ms=now_ms,
                    reset_result=pb.ResetResult(
                        observation=session.last_observation,
                        info={k: str(v) for k, v in (info or {}).items()},
                    ),
                )
                yield resp
                continue

            # Handle Step
            if req.HasField("step"):
                if session.env is None:
                    yield _error_response(
                        rid,
                        eid,
                        now_ms,
                        code=3,  # INVALID_ARGUMENT
                        message="Step received before Reset",
                    )
                    continue

                action = int(req.step.action)
                try:
                    obs, reward, terminated, truncated, info = session.env.step(action)
                except Exception as e:  # send error but keep stream alive
                    yield _error_response(
                        rid, eid, now_ms, code=13, message=f"step error: {e}"
                    )
                    continue

                session.step_index += 1
                session.last_observation = _to_list(obs)

                resp = pb.EnvResponse(
                    request_id=rid,
                    episode_id=session.episode_id or (eid or ""),
                    step_index=session.step_index,
                    server_time_ms=now_ms,
                    step_result=pb.StepResult(
                        observation=session.last_observation,
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        info={k: str(v) for k, v in (info or {}).items()},
                    ),
                )
                yield resp
                continue

            # Handle Close
            if req.HasField("close"):
                session.close_env()
                resp = pb.EnvResponse(
                    request_id=rid,
                    episode_id=eid or "",
                    step_index=0,
                    server_time_ms=now_ms,
                    closed=pb.Closed(message=req.close.reason or "closed by client"),
                )
                yield resp
                # We do NOT break; client may continue with a new Reset in same stream
                continue

            # Unknown/empty oneof -> error response
            yield _error_response(
                rid, eid, now_ms, code=3, message="No command in oneof"
            )


# -----------------
# Helpers
# -----------------


def _to_list(observation: object) -> list[float]:
    # Convert numpy arrays or sequences to plain Python list[float]
    try:
        import numpy as np  # local import to avoid hard dependency if not needed

        if isinstance(observation, np.ndarray):
            return observation.astype("float32").tolist()
    except Exception:
        pass
    # Fallback for list/tuple
    return [float(x) for x in observation]  # type: ignore[arg-type]


def _error_response(
    request_id: str,
    episode_id: str,
    server_time_ms: int,
    *,
    code: int,
    message: str,
) -> pb.EnvResponse:
    return pb.EnvResponse(
        request_id=request_id,
        episode_id=episode_id or "",
        step_index=0,
        server_time_ms=server_time_ms,
        error=pb.EnvError(code=code, message=message, details={}),
    )


def _gen_episode_id() -> str:
    # Simple time-based id; replace with UUID if needed
    return f"ep-{int(time.time() * 1000)}"


async def _serve() -> None:
    # Hardcode port per user preference (no argparse)
    address = "0.0.0.0:56111"  # adjust if needed
    server = grpc.aio.server()
    pb_grpc.add_CartPoleServiceServicer_to_server(CartPoleService(), server)
    server.add_insecure_port(address)
    await server.start()
    print(f"CartPoleService listening on {address}")
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(grace=None)


def main() -> None:
    # Run the asyncio server
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
