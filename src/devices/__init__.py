"""
KCake Devices - 设备适配层

支持任意平台：CPU, GPU (CUDA/MPS/Metal/Vulkan), NPU, TPU
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class Backend(Enum):
    """计算后端"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal
    VULKAN = "vulkan"
    NPU = "npu"
    TPU = "tpu"
    REMOTE = "remote"


@dataclass
class Device:
    """设备"""
    backend: Backend
    device_id: int
    name: str
    memory_total: int
    memory_available: int
    compute_units: int  # GPU cores or CPU threads
    numa_node: Optional[int] = None
    
    @property
    def memory_total_gb(self) -> float:
        return self.memory_total / (1024 ** 3)
    
    @property
    def memory_available_gb(self) -> float:
        return self.memory_available / (1024 ** 3)


class DeviceManager:
    """
    设备管理器
    
    探测和管理所有可用计算设备
    """
    
    def __init__(self):
        self.devices: Dict[str, Device] = {}
        self._detect_devices()
    
    def _detect_devices(self) -> None:
        """探测可用设备"""
        # CPU
        self._detect_cpu()
        
        # CUDA GPU
        self._detect_cuda()
        
        # Apple Metal
        self._detect_metal()
        
        # Vulkan
        self._detect_vulkan()
        
        # NPU
        self._detect_npu()
        
        logger.info(f"Detected {len(self.devices)} devices")
        for device_id, device in self.devices.items():
            logger.info(f"  {device_id}: {device.name} "
                       f"({device.backend.value}), "
                       f"Memory: {device.memory_available_gb:.1f} GB available")
    
    def _detect_cpu(self) -> None:
        """探测 CPU"""
        import os
        
        cpu_count = os.cpu_count() or 1
        
        # 尝试获取 CPU 内存
        memory_total = 0
        try:
            if hasattr(os, 'sysconf'):
                memory_total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        except Exception:
            memory_total = 8 * (1024 ** 3)  # 默认 8GB
        
        self.devices["cpu"] = Device(
            backend=Backend.CPU,
            device_id=0,
            name="CPU",
            memory_total=memory_total,
            memory_available=memory_total,
            compute_units=cpu_count,
        )
    
    def _detect_cuda(self) -> None:
        """探测 CUDA GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_total = props.total_memory
                    
                    self.devices[f"cuda:{i}"] = Device(
                        backend=Backend.CUDA,
                        device_id=i,
                        name=f"CUDA GPU {i} ({props.name})",
                        memory_total=memory_total,
                        memory_available=memory_total,
                        compute_units=props.multi_processor_count,
                    )
        except ImportError:
            logger.debug("PyTorch not available for CUDA detection")
        except Exception as e:
            logger.warning(f"CUDA detection failed: {e}")
    
    def _detect_metal(self) -> None:
        """探测 Apple Metal"""
        try:
            import torch
            if torch.backends.mps.is_available():
                self.devices["mps:0"] = Device(
                    backend=Backend.MPS,
                    device_id=0,
                    name="Apple Metal MPS",
                    memory_total=0,  # Metal 不直接暴露内存
                    memory_available=0,
                    compute_units=0,
                )
        except Exception as e:
            logger.debug(f"Metal detection failed: {e}")
    
    def _detect_vulkan(self) -> None:
        """探测 Vulkan (AMD/Intel)"""
        # Vulkan 检测需要 vulkaninfo 或 pyvulkan
        # 简化实现
        pass
    
    def _detect_npu(self) -> None:
        """探测 NPU (华为昇腾等)"""
        # NPU 检测需要特定 SDK
        # 简化实现
        pass
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """获取设备"""
        return self.devices.get(device_id)
    
    def get_all_devices(self) -> List[Device]:
        """获取所有设备"""
        return list(self.devices.values())
    
    def get_backend_priority(self, backend: Backend) -> int:
        """获取后端优先级 (越高越优先)"""
        priorities = {
            Backend.TPU: 100,
            Backend.CUDA: 90,
            Backend.MPS: 85,
            Backend.VULKAN: 80,
            Backend.NPU: 70,
            Backend.CPU: 50,
            Backend.REMOTE: 10,
        }
        return priorities.get(backend, 0)
    
    def select_optimal_device(
        self,
        required_memory: int,
        preferred_backends: Optional[List[Backend]] = None,
    ) -> Optional[Device]:
        """
        选择最优设备
        
        Args:
            required_memory: 所需内存 (bytes)
            preferred_backends: 优先的后端列表
        """
        candidates = []
        
        for device in self.devices.values():
            if device.memory_available < required_memory:
                continue
            
            if preferred_backends and device.backend not in preferred_backends:
                continue
            
            score = self.get_backend_priority(device.backend)
            score *= (device.memory_available / device.memory_total)
            candidates.append((device, score))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
