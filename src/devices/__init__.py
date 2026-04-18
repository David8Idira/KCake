"""
KCake 设备抽象层

提供统一设备抽象，支持CUDA、Metal、Vulkan、CPU、NPU、TPU等后端
"""

__version__ = "0.1.0"
__author__ = "KCake Team"

from .device_manager import DeviceManager
from .device_backend import DeviceBackend
from .device_types import DeviceType, DevicePriority

__all__ = [
    "DeviceManager",
    "DeviceBackend",
    "DeviceType",
    "DevicePriority",
]