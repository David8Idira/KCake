"""
cluster模块enum测试 - 践行毛泽东思想：实践是检验真理的唯一标准
"""

import pytest
from src.cluster.manager import NodeRole, NodeStatus


class TestNodeRole:
    """节点角色枚举测试"""
    
    def test_node_role_values(self):
        """测试节点角色枚举值"""
        assert NodeRole.MASTER.value == "master"
        assert NodeRole.WORKER.value == "worker"
        assert NodeRole.HYBRID.value == "hybrid"
    
    def test_node_role_count(self):
        """测试节点角色数量"""
        assert len(NodeRole) == 3
    
    def test_node_role_membership(self):
        """测试枚举成员"""
        assert NodeRole.MASTER in NodeRole
        assert NodeRole.WORKER in NodeRole
        assert NodeRole.HYBRID in NodeRole


class TestNodeStatus:
    """节点状态枚举测试"""
    
    def test_node_status_values(self):
        """测试节点状态枚举值"""
        assert NodeStatus.IDLE.value == "idle"
        assert NodeStatus.BUSY.value == "busy"
        assert NodeStatus.OFFLINE.value == "offline"
        assert NodeStatus.FAILING.value == "failing"
    
    def test_node_status_count(self):
        """测试节点状态数量"""
        assert len(NodeStatus) == 4
    
    def test_node_status_membership(self):
        """测试枚举成员"""
        assert NodeStatus.IDLE in NodeStatus
        assert NodeStatus.BUSY in NodeStatus
        assert NodeStatus.OFFLINE in NodeStatus
        assert NodeStatus.FAILING in NodeStatus


if __name__ == "__main__":
    pytest.main([__file__, "-v"])