"""
Expert Placer简单类型测试 - 践行毛泽东思想：实践是检验真理的唯一标准
"""

import pytest
from src.heterogeneous.expert_placer import PlacementResult, ExpertPlacer


class TestPlacementResult:
    """放置结果测试"""
    
    def test_placement_result_success(self):
        """测试成功放置结果"""
        result = PlacementResult(
            expert_id="expert-001",
            device_id="cuda:0",
            success=True,
            time_ms=10.5
        )
        
        assert result.expert_id == "expert-001"
        assert result.device_id == "cuda:0"
        assert result.success is True
        assert result.time_ms == 10.5
        assert result.error is None
    
    def test_placement_result_failure(self):
        """测试失败放置结果"""
        result = PlacementResult(
            expert_id="expert-002",
            device_id="",
            success=False,
            time_ms=0.0,
            error="Device not available"
        )
        
        assert result.success is False
        assert result.error == "Device not available"
    
    def test_placement_result_with_error(self):
        """测试带错误信息的放置结果"""
        result = PlacementResult(
            expert_id="expert-003",
            device_id="tpu:0",
            success=False,
            time_ms=5.0,
            error="Memory allocation failed"
        )
        
        assert result.success is False
        assert "failed" in result.error.lower()
    
    def test_placement_result_repr(self):
        """测试放置结果字符串表示"""
        result = PlacementResult(
            expert_id="test-expert",
            device_id="cpu:0",
            success=True,
            time_ms=3.0
        )
        
        repr_str = repr(result)
        assert "PlacementResult" in repr_str
        assert "test-expert" in repr_str


class TestExpertPlacer:
    """专家放置器测试"""
    
    def test_expert_placer_creation(self):
        """测试专家放置器创建"""
        placer = ExpertPlacer(scheduler=None)
        
        assert placer is not None
        assert placer.scheduler is None
    
    def test_expert_placer_with_scheduler(self):
        """测试带调度器的专家放置器"""
        # 创建mock调度器
        class MockScheduler:
            pass
        
        scheduler = MockScheduler()
        placer = ExpertPlacer(scheduler=scheduler)
        
        assert placer.scheduler is scheduler


if __name__ == "__main__":
    pytest.main([__file__, "-v"])