import io
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image


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
        # 创建环境时指定渲染模式，以便支持视频帧获取
        self.env = gym.make(env_id, render_mode=None)
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
        
        # 处理离散动作空间
        if act_space.__class__.__name__ == "Discrete":
            # 对于离散空间，取第一个值并转为整数
            action_idx = int(action_list[0]) if action_list else 0
            # 确保动作在有效范围内
            action_idx = max(0, min(action_idx, act_space.n - 1))
            return action_idx
            
        # 处理连续动作空间（Box）
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

    def render_frame(self, width: Optional[int] = None, height: Optional[int] = None, 
                     mode: str = "rgb_array") -> Dict[str, Any]:
        """渲染当前环境帧并返回图像数据。"""
        with self._lock:
            try:
                # 尝试渲染环境
                if hasattr(self.env, 'render'):
                    frame = self.env.render()
                    
                    if frame is None:
                        return {
                            "has_frame": False,
                            "image_data": b"",
                            "width": 0,
                            "height": 0,
                            "format": "png",
                            "timestamp": int(time.time() * 1000)
                        }
                    
                    # 转换为 numpy 数组
                    if isinstance(frame, np.ndarray):
                        # 如果是 RGB 数组
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            # 创建 PIL 图像
                            image = Image.fromarray(frame.astype(np.uint8))
                        else:
                            # 处理其他格式
                            image = Image.fromarray(frame)
                    else:
                        # 如果返回的不是数组，尝试直接使用
                        return {
                            "has_frame": False,
                            "image_data": b"",
                            "width": 0,
                            "height": 0,
                            "format": "png",
                            "timestamp": int(time.time() * 1000)
                        }
                    
                    # 调整图像大小（如果指定了宽度和高度）
                    if width is not None and height is not None and width > 0 and height > 0:
                        image = image.resize((width, height), Image.LANCZOS)
                    
                    # 转换为 PNG 字节流
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    image_data = buffer.getvalue()
                    
                    return {
                        "has_frame": True,
                        "image_data": image_data,
                        "width": image.width,
                        "height": image.height,
                        "format": "png",
                        "timestamp": int(time.time() * 1000)
                    }
                else:
                    # 环境不支持渲染
                    return {
                        "has_frame": False,
                        "image_data": b"",
                        "width": 0,
                        "height": 0,
                        "format": "png",
                        "timestamp": int(time.time() * 1000)
                    }
            except Exception as e:
                print(f"渲染帧时出错: {e}")
                return {
                    "has_frame": False,
                    "image_data": b"",
                    "width": 0,
                    "height": 0,
                    "format": "png",
                    "timestamp": int(time.time() * 1000)
                }
