"""
NUMA优化器简单类型测试 - 践行毛泽东思想：实践是检验真理的唯一标准
"""

import pytest
from typing import Dict, List
from src.heterogeneous.numa_optimizer import NUMANode, NUMAOptimizer


class TestNUMANode:
    """NUMA节点测试"""
    
    def test_numa_node_creation(self):
        """测试NUMA节点创建"""
        node = NUMANode(
            node_id=0,
            cpu_list=[0, 1, 2, 3],
            memory_bytes=64 * 1024 * 1024 * 1024,  # 64GB
            memory_available=32 * 1024 * 1024 * 1024,  # 32GB
            distance={0: 0, 1: 10, 2: 20, 3: 30}
        )
        
        assert node.node_id == 0
        assert node.cpu_list == [0, 1, 2, 3]
        assert node.memory_bytes == 64 * 1024 * 1024 * 1024
        assert node.memory_available == 32 * 1024 * 1024 * 1024
        assert node.distance[0] == 0
        assert node.distance[1] == 10
    
    def test_numa_node_single_cpu(self):
        """测试单CPU NUMA节点"""
        node = NUMANode(
            node_id=1,
            cpu_list=[4],
            memory_bytes=32 * 1024 * 1024 * 1024,  # 32GB
            memory_available=16 * 1024 * 1024 * 1024,  # 16GB
            distance={0: 10, 1: 0}
        )
        
        assert len(node.cpu_list) == 1
        assert node.cpu_list[0] == 4
    
    def test_numa_node_empty_distance(self):
        """测试空距离表"""
        node = NUMANode(
            node_id=2,
            cpu_list=[5, 6],
            memory_bytes=16 * 1024 * 1024 * 1024,
            memory_available=8 * 1024 * 1024 * 1024,
            distance={}
        )
        
        assert node.distance == {}


class TestNUMAOptimizer:
    """NUMA优化器测试"""
    
    def test_numa_optimizer_creation(self):
        """测试NUMA优化器创建"""
        optimizer = NUMAOptimizer()
        
        # is_available取决于系统是否支持NUMA
        assert isinstance(optimizer.is_available, bool)
        assert isinstance(optimizer.nodes, dict)
        # nodes可能包含自动检测到的NUMA节点
    
    def test_numa_optimizer_with_nodes(self):
        """测试带节点的NUMA优化器"""
        optimizer = NUMAOptimizer()
        
        node = NUMANode(
            node_id=99,  # 使用特殊ID避免冲突
            cpu_list=[0, 1],
            memory_bytes=64 * 1024 * 1024 * 1024,
            memory_available=32 * 1024 * 1024 * 1024,
            distance={99: 0}
        )
        optimizer.nodes[99] = node
        
        assert 99 in optimizer.nodes
        assert optimizer.nodes[99].node_id == 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])