import threading
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np


def _flatten(x: Any) -> List[float]:
    """将任意形状的数组展平成 1D float32 列表。"""
    return np.asarray(x, dtype=np.float32).ravel().tolist()


def _space_to_dict(space: Any) -> Dict[str, Any]:
    """将 Gymnasium 空间描述转为通用 dict，不依赖 protobuf。仅示范 Box。"""
    info: Dict[str, Any] = {
        "type": space.__class__.__name__,
    }
    if hasattr(space, "shape") and space.shape is not None:
        info["shape"] = list(space.shape)
    # 仅对 Box 暴露边界
    if space.__class__.__name__ == "Box":
        info["low"] = np.asarray(space.low, dtype=np.float32).ravel().tolist()
        info["high"] = np.asarray(space.high, dtype=np.float32).ravel().tolist()
    return info


class GrpcPlayer:
    """封装 Gymnasium 环境的仿真逻辑，线程安全；对上层返回纯 Python 数据。"""

    def __init__(self, env_id: str):
        self.env_id = env_id
        self.env = gym.make(env_id)
        self._lock = threading.Lock()

    # --------- 空间/规格 ---------
    def get_spec(self) -> Dict[str, Any]:
        obs_spec = _space_to_dict(self.env.observation_space)
        act_spec = _space_to_dict(self.env.action_space)
        return {
            "env_id": self.env_id,
            "observation": obs_spec,
            "action": act_spec,
        }

    # --------- 交互 API ---------
    def reset(self, seed: int | None) -> List[float]:
        """复位；seed<0 或 None 表示不设定。返回展平观测列表。"""
        actual_seed = None if seed is None or (isinstance(seed, int) and seed < 0) else seed
        with self._lock:
            obs, _info = self.env.reset(seed=actual_seed)
        return _flatten(obs)

    def _prepare_action(self, action_list: List[float]) -> Any:
        """将扁平动作恢复为环境动作空间形状并裁剪到边界。"""
        act_space = self.env.action_space
        if hasattr(act_space, "shape") and act_space.shape is not None:
            action = np.array(action_list, dtype=np.float32).reshape(act_space.shape)
            if hasattr(act_space, "low"):
                low = np.asarray(act_space.low, dtype=np.float32)
                high = np.asarray(act_space.high, dtype=np.float32)
                action = np.clip(action, low, high)
            return action
        # 退化：无形状空间，取第一个标量
        return float(action_list[0]) if action_list else 0.0

    def step(self, action_list: List[float]) -> Dict[str, Any]:
        """执行一步并返回结果字典。"""
        with self._lock:
            action = self._prepare_action(action_list)
            obs, reward, terminated, truncated, _info = self.env.step(action)

        return {
            "observation": _flatten(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def close(self) -> None:
        with self._lock:
            self.env.close()
