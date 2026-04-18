"""
KCake 集群管理测试

测试节点发现、集群管理、分片路由等功能
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import socket

from src.cluster.manager import (
    ClusterManager, ClusterConfig, NodeRole, NodeStatus,
    NodeInfo, ShardInfo
)
from src.cluster.discovery import mDNSDiscovery, ManualDiscovery, NodeDiscovery
from src.cluster.router import ShardRouter, RouteResult


class TestClusterConfig:
    """集群配置测试"""
    
    def test_config_creation(self):
        """测试配置创建"""
        config = ClusterConfig(
            cluster_key="test-key",
            master_host="192.168.1.1",
            master_port=8000,
            node_name="test-node",
            node_role=NodeRole.MASTER
        )
        
        assert config.cluster_key == "test-key"
        assert config.master_host == "192.168.1.1"
        assert config.master_port == 8000
        assert config.node_name == "test-node"
        assert config.node_role == NodeRole.MASTER


class TestNodeInfo:
    """节点信息测试"""
    
    def test_node_info_creation(self):
        """测试节点信息创建"""
        node = NodeInfo(
            node_id="node-1",
            name="Test Node",
            role=NodeRole.MASTER,
            host="192.168.1.1",
            port=8000,
            device_type="cuda",
            memory_total=32 * 1024**3,
            memory_available=20 * 1024**3,
            compute_score=0.9
        )
        
        assert node.node_id == "node-1"
        assert node.name == "Test Node"
        assert node.role == NodeRole.MASTER
        assert node.host == "192.168.1.1"
        assert node.port == 8000
        assert node.status == NodeStatus.IDLE


class TestClusterManager:
    """集群管理器测试"""
    
    @pytest.fixture
    def master_config(self):
        """创建Master配置"""
        return ClusterConfig(
            cluster_key="test-cluster",
            master_host="127.0.0.1",
            master_port=8000,
            node_name="master-node",
            node_role=NodeRole.MASTER
        )
    
    @pytest.fixture
    def worker_config(self):
        """创建Worker配置"""
        return ClusterConfig(
            cluster_key="test-cluster",
            master_host="127.0.0.1",
            master_port=8000,
            node_name="worker-node",
            node_role=NodeRole.WORKER
        )
    
    def test_cluster_manager_initialization_master(self, master_config):
        """测试Master节点初始化"""
        manager = ClusterManager(master_config)
        
        assert manager.is_master is True
        assert manager.is_running is False
        assert manager.self_info.role == NodeRole.MASTER
        assert manager.self_info.node_id is not None
    
    def test_cluster_manager_initialization_worker(self, worker_config):
        """测试Worker节点初始化"""
        manager = ClusterManager(worker_config)
        
        assert manager.is_master is False
        assert manager.self_info.role == NodeRole.WORKER
    
    def test_get_local_ip(self, master_config):
        """测试获取本地IP"""
        manager = ClusterManager(master_config)
        ip = manager._get_local_ip()
        
        assert ip is not None
        # IP应该是有效的格式
        assert len(ip.split(".")) == 4 or ip == "127.0.0.1"
    
    def test_register_shard(self, master_config):
        """测试注册分片"""
        manager = ClusterManager(master_config)
        
        shard = ShardInfo(
            shard_id=1,
            layer_start=0,
            layer_end=12,
            size_bytes=1024**3,
            checksum="abc123"
        )
        
        result = manager.register_shard(shard)
        assert result is True
        assert shard.shard_id in manager.shards
        assert shard.owner_node == manager.self_info.node_id
    
    def test_register_duplicate_shard(self, master_config):
        """测试注册重复分片"""
        manager = ClusterManager(master_config)
        
        shard = ShardInfo(
            shard_id=1,
            layer_start=0,
            layer_end=12,
            size_bytes=1024**3,
            checksum="abc123"
        )
        
        manager.register_shard(shard)
        result = manager.register_shard(shard)
        assert result is False
    
    def test_get_cluster_status(self, master_config):
        """测试获取集群状态"""
        manager = ClusterManager(master_config)
        manager.is_running = True
        
        status = manager.get_cluster_status()
        
        assert "is_master" in status
        assert "is_running" in status
        assert "total_nodes" in status
        assert "nodes" in status
        assert status["is_master"] is True
    
    def test_get_idle_nodes(self, master_config):
        """测试获取空闲节点"""
        manager = ClusterManager(master_config)
        
        idle_nodes = manager.get_idle_nodes()
        assert isinstance(idle_nodes, list)


class TestmDNSDiscovery:
    """mDNS发现测试"""
    
    @pytest.fixture
    def discovery(self):
        """创建mDNS发现实例"""
        return mDNSDiscovery(port=15353)
    
    def test_discovery_initialization(self, discovery):
        """测试发现初始化"""
        assert discovery.port == 15353
        assert discovery.is_running is False
    
    @pytest.mark.asyncio
    async def test_start_stop(self, discovery):
        """测试启动和停止"""
        await discovery.start()
        assert discovery.is_running is True
        
        await discovery.stop()
        assert discovery.is_running is False
    
    @pytest.mark.asyncio
    async def test_discover_nodes_empty(self, discovery):
        """测试发现空节点"""
        await discovery.start()
        try:
            nodes = await discovery.discover_nodes()
            assert isinstance(nodes, list)
        finally:
            await discovery.stop()


class TestManualDiscovery:
    """手动发现测试"""
    
    @pytest.fixture
    def discovery(self):
        """创建手动发现实例"""
        nodes = [
            {"node_id": "node-1", "name": "Node 1", "host": "192.168.1.1"},
            {"node_id": "node-2", "name": "Node 2", "host": "192.168.1.2"}
        ]
        return ManualDiscovery(nodes=nodes)
    
    def test_discovery_with_initial_nodes(self, discovery):
        """测试带初始节点的初始化"""
        nodes = discovery.nodes
        assert len(nodes) == 2
    
    @pytest.mark.asyncio
    async def test_discover_nodes(self, discovery):
        """测试发现节点"""
        nodes = await discovery.discover_nodes()
        assert len(nodes) == 2
    
    @pytest.mark.asyncio
    async def test_add_node(self, discovery):
        """测试添加节点"""
        new_node = {"node_id": "node-3", "name": "Node 3", "host": "192.168.1.3"}
        discovery.add_node(new_node)
        
        nodes = await discovery.discover_nodes()
        assert len(nodes) == 3
    
    @pytest.mark.asyncio
    async def test_remove_node(self, discovery):
        """测试移除节点"""
        result = discovery.remove_node("node-1")
        assert result is True
        
        nodes = await discovery.discover_nodes()
        assert len(nodes) == 1


class TestShardRouter:
    """分片路由器测试"""
    
    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟集群管理器"""
        manager = Mock()
        manager.self_info = Mock()
        manager.self_info.node_id = "self-node"
        manager.get_idle_nodes = Mock(return_value=[])
        return manager
    
    @pytest.fixture
    def router(self, mock_cluster_manager):
        """创建路由器实例"""
        return ShardRouter(mock_cluster_manager)
    
    def test_router_initialization(self, router):
        """测试路由器初始化"""
        assert router.num_shards == 8
        assert router.routing_strategy == "least_load"
    
    def test_router_configuration(self, router):
        """测试路由器配置"""
        router.configure(num_shards=16, layers_per_shard=4)
        assert router.num_shards == 16
        assert router.layers_per_shard == 4
    
    def test_calculate_shards_with_layers(self, router):
        """测试按层计算分片"""
        router.layers_per_shard = 4
        shards = router._calculate_shards(num_layers=16)
        
        assert len(shards) == 4
        assert shards[0] == (0, 4)
        assert shards[-1] == (12, 16)
    
    def test_calculate_shards_without_layers(self, router):
        """测试平均分片"""
        router.layers_per_shard = 0
        shards = router._calculate_shards(num_layers=16)
        
        assert len(shards) == 8  # 默认8个分片
    
    def test_get_stats(self, router):
        """测试获取统计"""
        stats = router.get_stats()
        
        assert "total_routes" in stats
        assert "local_routes" in stats
        assert "remote_routes" in stats
    
    def test_set_routing_strategy(self, router):
        """测试设置路由策略"""
        result = router.set_routing_strategy("round_robin")
        assert result is True
        assert router.routing_strategy == "round_robin"
    
    def test_set_invalid_routing_strategy(self, router):
        """测试设置无效路由策略"""
        result = router.set_routing_strategy("invalid")
        assert result is False


class TestRouteResult:
    """路由结果测试"""
    
    def test_route_result_creation(self):
        """测试创建路由结果"""
        result = RouteResult(
            shard_id=1,
            target_node="node-1",
            layer_range=(0, 4),
            is_local=True,
            estimated_latency_ms=10.5
        )
        
        assert result.shard_id == 1
        assert result.target_node == "node-1"
        assert result.layer_range == (0, 4)
        assert result.is_local is True
        assert result.estimated_latency_ms == 10.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])