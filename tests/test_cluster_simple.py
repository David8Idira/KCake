"""
集群管理简单类型测试 - 践行毛泽东思想：实践是检验真理的唯一标准

测试cluster模块中的dataclass等简单类型，快速提升覆盖率
"""

import pytest
from src.cluster.router import RouteResult
from src.cluster.manager import (
    NodeInfo, NodeRole, NodeStatus, ShardInfo, ClusterConfig
)


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


class TestNodeInfo:
    """节点信息测试"""
    
    def test_node_info_creation(self):
        """测试节点信息创建"""
        node = NodeInfo(
            node_id="node-001",
            name="Node 001",
            role=NodeRole.MASTER,
            host="192.168.1.100",
            port=8000,
            device_type="CUDA",
            memory_total=16 * 1024 * 1024 * 1024,  # 16GB
            memory_available=12 * 1024 * 1024 * 1024,  # 12GB
            compute_score=0.9
        )
        
        assert node.node_id == "node-001"
        assert node.name == "Node 001"
        assert node.role == NodeRole.MASTER
        assert node.host == "192.168.1.100"
        assert node.port == 8000
        assert node.device_type == "CUDA"
        assert node.memory_total == 16 * 1024 * 1024 * 1024
        assert node.memory_available == 12 * 1024 * 1024 * 1024
        assert node.compute_score == 0.9
        assert node.status == NodeStatus.IDLE
        assert node.last_heartbeat > 0
    
    def test_node_info_worker(self):
        """测试工作节点信息"""
        node = NodeInfo(
            node_id="node-002",
            name="Worker Node",
            role=NodeRole.WORKER,
            host="192.168.1.101",
            port=8001,
            device_type="CPU",
            memory_total=64 * 1024 * 1024 * 1024,  # 64GB
            memory_available=32 * 1024 * 1024 * 1024,  # 32GB
            compute_score=0.6,
            status=NodeStatus.BUSY
        )
        
        assert node.role == NodeRole.WORKER
        assert node.status == NodeStatus.BUSY
    
    def test_node_info_repr(self):
        """测试节点信息字符串表示"""
        node = NodeInfo(
            node_id="test-node",
            name="Test Node",
            role=NodeRole.WORKER,
            host="localhost",
            port=8080,
            device_type="TEST",
            memory_total=1024,
            memory_available=512,
            compute_score=0.5
        )
        
        repr_str = repr(node)
        assert "NodeInfo" in repr_str
        assert "test-node" in repr_str


class TestShardInfo:
    """分片信息测试"""
    
    def test_shard_info_creation(self):
        """测试分片信息创建"""
        shard = ShardInfo(
            shard_id=1,
            layer_start=0,
            layer_end=8,
            size_bytes=1024 * 1024 * 100,  # 100MB
            checksum="abc123def456"
        )
        
        assert shard.shard_id == 1
        assert shard.layer_start == 0
        assert shard.layer_end == 8
        assert shard.size_bytes == 1024 * 1024 * 100
        assert shard.checksum == "abc123def456"
        assert shard.owner_node is None
        assert shard.cached is False
        assert shard.access_count == 0
    
    def test_shard_info_with_owner(self):
        """测试有所有者的分片信息"""
        shard = ShardInfo(
            shard_id=2,
            layer_start=8,
            layer_end=16,
            size_bytes=1024 * 1024 * 200,  # 200MB
            checksum="def456ghi789",
            owner_node="node-001",
            cached=True,
            access_count=100
        )
        
        assert shard.owner_node == "node-001"
        assert shard.cached is True
        assert shard.access_count == 100
    
    def test_shard_info_repr(self):
        """测试分片信息字符串表示"""
        shard = ShardInfo(
            shard_id=3,
            layer_start=0,
            layer_end=1,
            size_bytes=1024,
            checksum="test"
        )
        
        repr_str = repr(shard)
        assert "ShardInfo" in repr_str
        assert "shard_id=3" in repr_str


class TestClusterConfig:
    """集群配置测试"""
    
    def test_cluster_config_creation(self):
        """测试集群配置创建"""
        config = ClusterConfig(
            cluster_key="kcake-cluster-001",
            master_host="192.168.1.100",
            master_port=8000,
            node_name="worker-001",
            node_role=NodeRole.WORKER
        )
        
        assert config.cluster_key == "kcake-cluster-001"
        assert config.master_host == "192.168.1.100"
        assert config.master_port == 8000
        assert config.node_name == "worker-001"
        assert config.node_role == NodeRole.WORKER
        assert config.heartbeat_interval == 10
        assert config.heartbeat_timeout == 30
        assert config.max_retry == 3
    
    def test_cluster_config_master(self):
        """测试主节点配置"""
        config = ClusterConfig(
            cluster_key="kcake-master",
            master_host="127.0.0.1",
            master_port=9000,
            node_name="master-node",
            node_role=NodeRole.MASTER,
            heartbeat_interval=5,
            heartbeat_timeout=15,
            max_retry=5
        )
        
        assert config.node_role == NodeRole.MASTER
        assert config.heartbeat_interval == 5
        assert config.heartbeat_timeout == 15
        assert config.max_retry == 5
    
    def test_cluster_config_repr(self):
        """测试集群配置字符串表示"""
        config = ClusterConfig(
            cluster_key="test-cluster",
            master_host="localhost",
            master_port=8080,
            node_name="test-node"
        )
        
        repr_str = repr(config)
        assert "ClusterConfig" in repr_str
        assert "test-cluster" in repr_str


class TestNodeRole:
    """节点角色测试"""
    
    def test_enum_values(self):
        """测试枚举值"""
        assert NodeRole.MASTER.value == "master"
        assert NodeRole.WORKER.value == "worker"
        assert NodeRole.HYBRID.value == "hybrid"
    
    def test_enum_membership(self):
        """测试枚举成员"""
        assert NodeRole.MASTER in NodeRole
        assert NodeRole.WORKER in NodeRole
        assert NodeRole.HYBRID in NodeRole


class TestNodeStatus:
    """节点状态测试"""
    
    def test_enum_values(self):
        """测试枚举值"""
        assert NodeStatus.IDLE.value == "idle"
        assert NodeStatus.BUSY.value == "busy"
        assert NodeStatus.OFFLINE.value == "offline"
        assert NodeStatus.FAILING.value == "failing"
    
    def test_enum_membership(self):
        """测试枚举成员"""
        assert NodeStatus.IDLE in NodeStatus
        assert NodeStatus.BUSY in NodeStatus
        assert NodeStatus.OFFLINE in NodeStatus
        assert NodeStatus.FAILING in NodeStatus


if __name__ == "__main__":
    """践行毛泽东思想：实践是检验真理的唯一标准"""
    print("开始执行集群管理简单类型测试...")
    pytest.main([__file__, "-v"])