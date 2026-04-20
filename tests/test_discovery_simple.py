"""
集群发现简单测试 - 践行毛泽东思想：实践是检验真理的唯一标准
"""

import pytest
from abc import ABC
from src.cluster.discovery import NodeDiscovery, mDNSDiscovery


class TestNodeDiscovery:
    """节点发现基类测试"""
    
    def test_node_discovery_is_abc(self):
        """测试NodeDiscovery是抽象基类"""
        assert issubclass(NodeDiscovery, ABC)
    
    def test_node_discovery_has_abstract_methods(self):
        """测试NodeDiscovery有抽象方法"""
        # start方法是抽象的
        assert hasattr(NodeDiscovery, 'start')
        # stop方法是抽象的
        assert hasattr(NodeDiscovery, 'stop')
        # discover_nodes方法是抽象的
        assert hasattr(NodeDiscovery, 'discover_nodes')
        # announce方法是抽象的
        assert hasattr(NodeDiscovery, 'announce')


class TestmDNSDiscovery:
    """mDNS发现实现测试"""
    
    def test_mdns_discovery_creation(self):
        """测试mDNSDiscovery创建"""
        discovery = mDNSDiscovery(port=5353)
        assert discovery is not None
        assert discovery.port == 5353
    
    def test_mdns_discovery_service_type(self):
        """测试mDNS服务类型"""
        assert mDNSDiscovery.SERVICE_TYPE == "_kcake._tcp"
        assert mDNSDiscovery.SERVICE_NAME == "KCake Cluster"
    
    def test_mdns_discovery_inherits(self):
        """测试mDNSDiscovery继承NodeDiscovery"""
        assert issubclass(mDNSDiscovery, NodeDiscovery)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])