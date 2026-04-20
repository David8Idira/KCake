"""
cluster路由简单测试 - 践行毛泽东思想：实践是检验真理的唯一标准
"""

import pytest
from src.cluster.router import RouteResult


class TestRouteResult:
    """路由结果测试"""
    
    def test_route_result_creation(self):
        """测试路由结果创建"""
        result = RouteResult(
            shard_id=1,
            target_node="node-001",
            layer_range=(0, 8),
            is_local=True,
            estimated_latency_ms=25.5
        )
        
        assert result.shard_id == 1
        assert result.target_node == "node-001"
        assert result.layer_range == (0, 8)
        assert result.is_local is True
        assert result.estimated_latency_ms == 25.5
    
    def test_route_result_remote(self):
        """测试远程路由结果"""
        result = RouteResult(
            shard_id=2,
            target_node="node-002",
            layer_range=(8, 16),
            is_local=False,
            estimated_latency_ms=150.0
        )
        
        assert result.shard_id == 2
        assert result.target_node == "node-002"
        assert result.layer_range == (8, 16)
        assert result.is_local is False
        assert result.estimated_latency_ms == 150.0
    
    def test_route_result_repr(self):
        """测试路由结果字符串表示"""
        result = RouteResult(
            shard_id=3,
            target_node="test-node",
            layer_range=(0, 1),
            is_local=True,
            estimated_latency_ms=10.0
        )
        
        repr_str = repr(result)
        assert "RouteResult" in repr_str
        assert "shard_id=3" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])