"""
异构模块枚举类型测试 - 践行毛泽东思想：实践是检验真理的唯一标准

测试heterogeneous模块中的枚举类型和简单dataclass
"""

import pytest
from enum import Enum
from src.heterogeneous.scheduler import (
    ExpertType, DeviceType, ExpertInfo, DeviceInfo, ScheduleResult
)


class TestExpertType:
    """专家类型枚举测试"""
    
    def test_expert_type_values(self):
        """测试专家类型枚举值"""
        assert ExpertType.HOT.value == "hot"
        assert ExpertType.WARM.value == "warm"
        assert ExpertType.COLD.value == "cold"
    
    def test_expert_type_count(self):
        """测试专家类型数量"""
        assert len(ExpertType) == 3
    
    def test_expert_type_membership(self):
        """测试枚举成员"""
        assert ExpertType.HOT in ExpertType
        assert ExpertType.WARM in ExpertType
        assert ExpertType.COLD in ExpertType


class TestDeviceType:
    """设备类型枚举测试"""
    
    def test_device_type_values(self):
        """测试设备类型枚举值"""
        assert DeviceType.TPU.value == "tpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.METAL.value == "metal"
        assert DeviceType.VULKAN.value == "vulkan"
        assert DeviceType.NPU.value == "npu"
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.DISK.value == "disk"
        assert DeviceType.REMOTE.value == "remote"
    
    def test_device_type_count(self):
        """测试设备类型数量"""
        assert len(DeviceType) == 8
    
    def test_device_type_membership(self):
        """测试枚举成员"""
        assert DeviceType.CUDA in DeviceType
        assert DeviceType.CPU in DeviceType
        assert DeviceType.TPU in DeviceType


class TestExpertInfo:
    """专家信息测试"""
    
    def test_expert_info_creation(self):
        """测试专家信息创建"""
        expert = ExpertInfo(
            expert_id="expert-001",
            name="Expert One",
            layer_indices=[0, 1, 2, 3],
            size_bytes=1024 * 1024 * 100  # 100MB
        )
        
        assert expert.expert_id == "expert-001"
        assert expert.name == "Expert One"
        assert expert.layer_indices == [0, 1, 2, 3]
        assert expert.size_bytes == 1024 * 1024 * 100
        assert expert.expert_type == ExpertType.WARM  # default
        assert expert.call_frequency == 0.0  # default
    
    def test_expert_info_hot(self):
        """测试高频专家信息"""
        expert = ExpertInfo(
            expert_id="expert-hot",
            name="Hot Expert",
            layer_indices=[0, 1],
            size_bytes=1024 * 1024 * 200,  # 200MB
            expert_type=ExpertType.HOT,
            call_frequency=0.9
        )
        
        assert expert.expert_type == ExpertType.HOT
        assert expert.call_frequency == 0.9
    
    def test_expert_info_cold(self):
        """测试低频专家信息"""
        expert = ExpertInfo(
            expert_id="expert-cold",
            name="Cold Expert",
            layer_indices=[12, 13, 14, 15, 16],
            size_bytes=1024 * 1024 * 50,  # 50MB
            expert_type=ExpertType.COLD,
            call_frequency=0.1
        )
        
        assert expert.expert_type == ExpertType.COLD
        assert expert.call_frequency == 0.1


class TestDeviceInfo:
    """设备信息测试"""
    
    def test_device_info_creation(self):
        """测试设备信息创建"""
        device = DeviceInfo(
            device_id="cuda:0",
            device_type=DeviceType.CUDA,
            name="NVIDIA GPU 0",
            memory_total=16 * 1024 * 1024 * 1024,  # 16GB
            memory_available=12 * 1024 * 1024 * 1024,  # 12GB
            compute_score=1.0,
            bandwidth_score=0.9
        )
        
        assert device.device_id == "cuda:0"
        assert device.device_type == DeviceType.CUDA
        assert device.name == "NVIDIA GPU 0"
        assert device.memory_total == 16 * 1024 * 1024 * 1024
        assert device.memory_available == 12 * 1024 * 1024 * 1024
        assert device.compute_score == 1.0
        assert device.bandwidth_score == 0.9
        assert device.is_available is True  # default
        assert device.current_load == 0.0  # default
    
    def test_device_info_cpu(self):
        """测试CPU设备信息"""
        device = DeviceInfo(
            device_id="cpu:0",
            device_type=DeviceType.CPU,
            name="CPU 0",
            memory_total=64 * 1024 * 1024 * 1024,  # 64GB
            memory_available=32 * 1024 * 1024 * 1024,  # 32GB
            compute_score=0.5,
            bandwidth_score=0.8,
            is_available=False,
            current_load=0.7
        )
        
        assert device.device_type == DeviceType.CPU
        assert device.compute_score == 0.5
        assert device.is_available is False
        assert device.current_load == 0.7
    
    def test_device_info_tpu(self):
        """测试TPU设备信息"""
        device = DeviceInfo(
            device_id="tpu:0",
            device_type=DeviceType.TPU,
            name="TPU 0",
            memory_total=8 * 1024 * 1024 * 1024,  # 8GB
            memory_available=6 * 1024 * 1024 * 1024,  # 6GB
            compute_score=0.8,
            bandwidth_score=0.7
        )
        
        assert device.device_type == DeviceType.TPU
        assert device.compute_score == 0.8


class TestScheduleResult:
    """调度结果测试"""
    
    def test_schedule_result_creation(self):
        """测试调度结果创建"""
        result = ScheduleResult(
            expert_id="expert-001",
            target_device="cuda:0",
            migration_needed=False,
            estimated_time_ms=10.5,
            reason="optimal placement"
        )
        
        assert result.expert_id == "expert-001"
        assert result.target_device == "cuda:0"
        assert result.migration_needed is False
        assert result.estimated_time_ms == 10.5
        assert result.reason == "optimal placement"
    
    def test_schedule_result_migration(self):
        """测试需要迁移的调度结果"""
        result = ScheduleResult(
            expert_id="expert-002",
            target_device="cpu:0",
            migration_needed=True,
            estimated_time_ms=50.0,
            reason="GPU memory full"
        )
        
        assert result.migration_needed is True
        assert result.estimated_time_ms == 50.0
    
    def test_schedule_result_repr(self):
        """测试调度结果字符串表示"""
        result = ScheduleResult(
            expert_id="test-expert",
            target_device="tpu:0",
            migration_needed=False,
            estimated_time_ms=5.0,
            reason="test"
        )
        
        repr_str = repr(result)
        assert "ScheduleResult" in repr_str
        assert "test-expert" in repr_str


if __name__ == "__main__":
    """践行毛泽东思想：实践是检验真理的唯一标准"""
    print("开始执行异构模块枚举类型测试...")
    pytest.main([__file__, "-v"])