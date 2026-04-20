"""
异构调度器简单类型测试 - 践行毛泽东思想：实践是检验真理的唯一标准

测试enum和dataclass等简单类型，快速提升覆盖率
"""

import pytest
import time
from src.heterogeneous.scheduler import (
    ExpertType,
    DeviceType,
    ExpertInfo,
    DeviceInfo,
    ScheduleResult
)


class TestExpertType:
    """专家类型测试"""
    
    def test_enum_values(self):
        """测试枚举值"""
        assert ExpertType.HOT.value == "hot"
        assert ExpertType.WARM.value == "warm"
        assert ExpertType.COLD.value == "cold"
    
    def test_enum_membership(self):
        """测试枚举成员"""
        assert ExpertType.HOT in ExpertType
        assert ExpertType.WARM in ExpertType
        assert ExpertType.COLD in ExpertType
    
    def test_enum_iteration(self):
        """测试枚举迭代"""
        types = list(ExpertType)
        assert len(types) == 3
        assert ExpertType.HOT in types
        assert ExpertType.WARM in types
        assert ExpertType.COLD in types
    
    def test_enum_comparison(self):
        """测试枚举比较"""
        assert ExpertType.HOT == ExpertType.HOT
        assert ExpertType.HOT != ExpertType.WARM


class TestDeviceType:
    """设备类型测试"""
    
    def test_enum_values(self):
        """测试枚举值"""
        assert DeviceType.TPU.value == "tpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.METAL.value == "metal"
        assert DeviceType.VULKAN.value == "vulkan"
        assert DeviceType.NPU.value == "npu"
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.DISK.value == "disk"
        assert DeviceType.REMOTE.value == "remote"
    
    def test_enum_membership(self):
        """测试枚举成员"""
        assert DeviceType.TPU in DeviceType
        assert DeviceType.CUDA in DeviceType
        assert DeviceType.CPU in DeviceType
    
    def test_enum_count(self):
        """测试枚举数量"""
        types = list(DeviceType)
        assert len(types) >= 7  # 至少有7种设备类型


class TestExpertInfo:
    """专家信息测试"""
    
    def test_expert_info_creation(self):
        """测试专家信息创建"""
        expert = ExpertInfo(
            expert_id="expert-001",
            name="attention-expert",
            layer_indices=[1, 2, 3],
            size_bytes=1024 * 1024,  # 1MB
            call_frequency=0.8,
            expert_type=ExpertType.HOT
        )
        
        assert expert.expert_id == "expert-001"
        assert expert.name == "attention-expert"
        assert expert.layer_indices == [1, 2, 3]
        assert expert.size_bytes == 1024 * 1024
        assert expert.call_frequency == 0.8
        assert expert.expert_type == ExpertType.HOT
        assert expert.current_device is None
        assert isinstance(expert.last_updated, float)
    
    def test_expert_info_defaults(self):
        """测试专家信息默认值"""
        expert = ExpertInfo(
            expert_id="expert-002",
            name="test-expert",
            layer_indices=[4],
            size_bytes=512 * 1024
        )
        
        assert expert.call_frequency == 0.0
        assert expert.expert_type == ExpertType.WARM
        assert expert.current_device is None
        assert expert.last_updated > 0
    
    def test_expert_info_with_device(self):
        """测试带设备的专家信息"""
        expert = ExpertInfo(
            expert_id="expert-003",
            name="gpu-expert",
            layer_indices=[5, 6],
            size_bytes=2048 * 1024,
            current_device="cuda:0"
        )
        
        assert expert.current_device == "cuda:0"
    
    def test_expert_info_repr(self):
        """测试专家信息字符串表示"""
        expert = ExpertInfo(
            expert_id="test-id",
            name="test-name",
            layer_indices=[1],
            size_bytes=1024
        )
        
        repr_str = repr(expert)
        assert "ExpertInfo" in repr_str
        assert "test-id" in repr_str


class TestDeviceInfo:
    """设备信息测试"""
    
    def test_device_info_creation(self):
        """测试设备信息创建"""
        device = DeviceInfo(
            device_id="device-001",
            device_type=DeviceType.CUDA,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024 * 1024 * 1024,  # 24GB
            memory_available=20 * 1024 * 1024 * 1024,  # 20GB
            compute_score=0.95,
            bandwidth_score=0.9
        )
        
        assert device.device_id == "device-001"
        assert device.device_type == DeviceType.CUDA
        assert device.name == "NVIDIA RTX 4090"
        assert device.memory_total == 24 * 1024 * 1024 * 1024
        assert device.memory_available == 20 * 1024 * 1024 * 1024
        assert device.compute_score == 0.95
        assert device.bandwidth_score == 0.9
        assert device.is_available is True
        assert device.current_load == 0.0
    
    def test_device_info_defaults(self):
        """测试设备信息默认值"""
        device = DeviceInfo(
            device_id="device-002",
            device_type=DeviceType.CPU,
            name="Intel Xeon",
            memory_total=64 * 1024 * 1024 * 1024,
            memory_available=32 * 1024 * 1024 * 1024,
            compute_score=0.6,
            bandwidth_score=0.5
        )
        
        assert device.is_available is True
        assert device.current_load == 0.0
    
    def test_device_info_unavailable(self):
        """测试不可用设备"""
        device = DeviceInfo(
            device_id="device-003",
            device_type=DeviceType.DISK,
            name="SSD Storage",
            memory_total=1024 * 1024 * 1024 * 1024,  # 1TB
            memory_available=500 * 1024 * 1024 * 1024,
            compute_score=0.1,
            bandwidth_score=0.3,
            is_available=False,
            current_load=0.8
        )
        
        assert device.is_available is False
        assert device.current_load == 0.8
    
    def test_device_info_repr(self):
        """测试设备信息字符串表示"""
        device = DeviceInfo(
            device_id="test-device",
            device_type=DeviceType.CPU,
            name="Test CPU",
            memory_total=1024,
            memory_available=512,
            compute_score=0.5,
            bandwidth_score=0.5
        )
        
        repr_str = repr(device)
        assert "DeviceInfo" in repr_str
        assert "test-device" in repr_str


class TestScheduleResult:
    """调度结果测试"""
    
    def test_schedule_result_creation(self):
        """测试调度结果创建"""
        result = ScheduleResult(
            expert_id="expert-001",
            target_device="cuda:0",
            migration_needed=True,
            estimated_time_ms=150.5,
            reason="Load balancing"
        )
        
        assert result.expert_id == "expert-001"
        assert result.target_device == "cuda:0"
        assert result.migration_needed is True
        assert result.estimated_time_ms == 150.5
        assert result.reason == "Load balancing"
    
    def test_schedule_result_no_migration(self):
        """测试无需迁移的调度结果"""
        result = ScheduleResult(
            expert_id="expert-002",
            target_device="cpu:0",
            migration_needed=False,
            estimated_time_ms=0.0,
            reason="Already on optimal device"
        )
        
        assert result.migration_needed is False
        assert result.estimated_time_ms == 0.0
    
    def test_schedule_result_repr(self):
        """测试调度结果字符串表示"""
        result = ScheduleResult(
            expert_id="test-expert",
            target_device="test-device",
            migration_needed=True,
            estimated_time_ms=100.0,
            reason="test reason"
        )
        
        repr_str = repr(result)
        assert "ScheduleResult" in repr_str
        assert "test-expert" in repr_str


if __name__ == "__main__":
    """践行毛泽东思想：实践是检验真理的唯一标准"""
    print("开始执行异构调度器简单类型测试...")
    pytest.main([__file__, "-v"])