"""
KCake 异构调度器测试

测试调度器、设备管理、专家放置等功能
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock

from src.heterogeneous.scheduler import (
    HeteroScheduler, ExpertType, DeviceType, 
    ExpertInfo, DeviceInfo, ScheduleResult
)
from src.heterogeneous.numa_optimizer import NUMAOptimizer, NUMANode
from src.heterogeneous.expert_placer import ExpertPlacer, PlacementResult


class TestHeteroScheduler:
    """异构调度器测试"""
    
    @pytest.fixture
    def scheduler(self):
        """创建调度器实例"""
        return HeteroScheduler(config={
            "hot_threshold": 0.8,
            "cold_threshold": 0.3
        })
    
    @pytest.fixture
    def sample_device(self):
        """创建示例设备"""
        return DeviceInfo(
            device_id="device-1",
            device_type=DeviceType.CUDA,
            name="NVIDIA RTX 3090",
            memory_total=24 * 1024**3,  # 24GB
            memory_available=20 * 1024**3,
            compute_score=0.9,
            bandwidth_score=0.8
        )
    
    @pytest.fixture
    def sample_expert(self):
        """创建示例专家"""
        return ExpertInfo(
            expert_id="expert-1",
            name="attention-layer",
            layer_indices=[0, 1, 2],
            size_bytes=1024**3,  # 1GB
            call_frequency=0.5
        )
    
    def test_scheduler_initialization(self, scheduler):
        """测试调度器初始化"""
        assert scheduler.hot_threshold == 0.8
        assert scheduler.cold_threshold == 0.3
        assert len(scheduler.experts) == 0
        assert len(scheduler.devices) == 0
    
    def test_register_device(self, scheduler, sample_device):
        """测试设备注册"""
        result = scheduler.register_device(sample_device)
        assert result is True
        assert sample_device.device_id in scheduler.devices
    
    def test_register_duplicate_device(self, scheduler, sample_device):
        """测试重复设备注册"""
        scheduler.register_device(sample_device)
        result = scheduler.register_device(sample_device)
        assert result is False
    
    def test_unregister_device(self, scheduler, sample_device):
        """测试设备注销"""
        scheduler.register_device(sample_device)
        result = scheduler.unregister_device(sample_device.device_id)
        assert result is True
        assert sample_device.device_id not in scheduler.devices
    
    def test_register_expert(self, scheduler, sample_expert):
        """测试专家注册"""
        result = scheduler.register_expert(sample_expert)
        assert result is True
        assert sample_expert.expert_id in scheduler.experts
    
    def test_expert_classification(self, scheduler):
        """测试专家分类"""
        # Hot专家
        hot_expert = ExpertInfo(
            expert_id="hot-1",
            name="hot-expert",
            layer_indices=[0],
            size_bytes=100,
            call_frequency=0.9
        )
        scheduler.register_expert(hot_expert)
        assert hot_expert.expert_type == ExpertType.HOT
        
        # Warm专家
        warm_expert = ExpertInfo(
            expert_id="warm-1",
            name="warm-expert",
            layer_indices=[0],
            size_bytes=100,
            call_frequency=0.5
        )
        scheduler.register_expert(warm_expert)
        assert warm_expert.expert_type == ExpertType.WARM
        
        # Cold专家
        cold_expert = ExpertInfo(
            expert_id="cold-1",
            name="cold-expert",
            layer_indices=[0],
            size_bytes=100,
            call_frequency=0.1
        )
        scheduler.register_expert(cold_expert)
        assert cold_expert.expert_type == ExpertType.COLD
    
    def test_update_expert_frequency(self, scheduler, sample_expert):
        """测试更新专家频率"""
        scheduler.register_expert(sample_expert)
        
        # 更新为高频
        result = scheduler.update_expert_frequency("expert-1", 0.9)
        assert result is True
        assert scheduler.experts["expert-1"].call_frequency == 0.9
        assert scheduler.experts["expert-1"].expert_type == ExpertType.HOT
        
        # 更新为低频
        result = scheduler.update_expert_frequency("expert-1", 0.1)
        assert result is True
        assert scheduler.experts["expert-1"].expert_type == ExpertType.COLD
    
    @pytest.mark.asyncio
    async def test_schedule_expert_with_no_device(self, scheduler, sample_expert):
        """测试无可用设备时的调度"""
        scheduler.register_expert(sample_expert)
        
        result = await scheduler.schedule_expert("expert-1")
        assert result is not None
        assert result.target_device == ""
    
    @pytest.mark.asyncio
    async def test_schedule_expert_with_device(self, scheduler, sample_expert, sample_device):
        """测试有设备时的调度"""
        scheduler.register_device(sample_device)
        scheduler.register_expert(sample_expert)
        
        result = await scheduler.schedule_expert("expert-1")
        assert result is not None
        assert result.target_device == sample_device.device_id
    
    def test_get_expert_placement(self, scheduler, sample_expert, sample_device):
        """测试获取专家放置情况"""
        scheduler.register_device(sample_device)
        scheduler.register_expert(sample_expert)
        sample_expert.current_device = sample_device.device_id
        
        placement = scheduler.get_expert_placement()
        assert "expert-1" in placement
        assert placement["expert-1"] == sample_device.device_id
    
    def test_get_device_utilization(self, scheduler, sample_device, sample_expert):
        """测试获取设备利用率"""
        scheduler.register_device(sample_device)
        scheduler.register_expert(sample_expert)
        sample_expert.current_device = sample_device.device_id
        
        utilization = scheduler.get_device_utilization()
        assert "device-1" in utilization
        assert utilization["device-1"]["experts_count"] == 1
    
    def test_get_stats(self, scheduler):
        """测试获取统计信息"""
        stats = scheduler.get_stats()
        assert "total_schedules" in stats
        assert "successful_migrations" in stats
        assert stats["total_schedules"] == 0


class TestNUMAOptimizer:
    """NUMA优化器测试"""
    
    @pytest.fixture
    def optimizer(self):
        """创建优化器实例"""
        return NUMAOptimizer()
    
    def test_optimizer_initialization(self, optimizer):
        """测试优化器初始化"""
        assert optimizer is not None
    
    def test_get_info(self, optimizer):
        """测试获取NUMA信息"""
        info = optimizer.get_info()
        assert "is_available" in info


class TestExpertPlacer:
    """专家放置器测试"""
    
    @pytest.fixture
    def mock_scheduler(self):
        """创建模拟调度器"""
        scheduler = Mock()
        scheduler.devices = {}
        scheduler.experts = {}
        return scheduler
    
    @pytest.fixture
    def placer(self, mock_scheduler):
        """创建放置器实例"""
        return ExpertPlacer(scheduler=mock_scheduler)
    
    def test_placer_initialization(self, placer):
        """测试放置器初始化"""
        assert placer is not None
        assert len(placer.placements) == 0
    
    def test_get_placement_info(self, placer):
        """测试获取放置信息"""
        info = placer.get_placement_info()
        assert "total_placements" in info
        assert "placements" in info
        assert "stats" in info
    
    def test_get_expert_device_empty(self, placer):
        """测试获取未放置的专家设备"""
        device = placer.get_expert_device("nonexistent")
        assert device is None
    
    def test_get_device_experts_empty(self, placer):
        """测试获取空设备的专家"""
        experts = placer.get_device_experts("device-1")
        assert len(experts) == 0


class TestExpertInfo:
    """专家信息测试"""
    
    def test_expert_info_creation(self):
        """测试创建专家信息"""
        expert = ExpertInfo(
            expert_id="exp-1",
            name="test-expert",
            layer_indices=[0, 1, 2],
            size_bytes=1024,
            call_frequency=0.5
        )
        
        assert expert.expert_id == "exp-1"
        assert expert.name == "test-expert"
        assert expert.layer_indices == [0, 1, 2]
        assert expert.size_bytes == 1024
        assert expert.call_frequency == 0.5
        assert expert.expert_type == ExpertType.WARM


class TestDeviceInfo:
    """设备信息测试"""
    
    def test_device_info_creation(self):
        """测试创建设备信息"""
        device = DeviceInfo(
            device_id="dev-1",
            device_type=DeviceType.CUDA,
            name="NVIDIA GPU",
            memory_total=24 * 1024**3,
            memory_available=20 * 1024**3,
            compute_score=0.9,
            bandwidth_score=0.8
        )
        
        assert device.device_id == "dev-1"
        assert device.device_type == DeviceType.CUDA
        assert device.memory_total == 24 * 1024**3
        assert device.is_available is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])