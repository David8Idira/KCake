#!/usr/bin/env python3
"""
KCake 基础测试运行器

不依赖torch的测试
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, '/root/workspace/kcake')

def test_kv_cache():
    """测试KV缓存"""
    print("测试KV缓存...")
    
    import numpy as np
    from src.core.kv_cache import KVCache
    
    cache = KVCache(max_size_mb=10, max_entries=100)
    
    # 测试存入
    test_data = np.random.rand(100).astype(np.float32)
    cache.put("key1", test_data)
    
    # 测试取出
    retrieved = cache.get("key1")
    assert retrieved is not None, "缓存未命中"
    np.testing.assert_array_equal(retrieved, test_data)
    
    # 测试统计
    stats = cache.get_stats()
    assert stats["hits"] == 1
    
    print("✅ KV缓存测试通过")
    return True


def test_scheduler():
    """测试调度器"""
    print("测试异构调度器...")
    
    from src.heterogeneous.scheduler import HeteroScheduler, DeviceType, ExpertType, DeviceInfo, ExpertInfo
    
    scheduler = HeteroScheduler()
    
    # 注册设备
    device = DeviceInfo(
        device_id="cuda-1",
        device_type=DeviceType.CUDA,
        name="NVIDIA RTX 3090",
        memory_total=24 * 1024**3,
        memory_available=20 * 1024**3,
        compute_score=0.9,
        bandwidth_score=0.8
    )
    scheduler.register_device(device)
    
    # 注册专家
    expert = ExpertInfo(
        expert_id="exp-1",
        name="attention-layer",
        layer_indices=[0, 1, 2],
        size_bytes=1024**3,
        call_frequency=0.5
    )
    scheduler.register_expert(expert)
    
    # 验证
    assert "cuda-1" in scheduler.devices
    assert "exp-1" in scheduler.experts
    
    # 测试专家分类
    scheduler.update_expert_frequency("exp-1", 0.9)
    assert expert.expert_type == ExpertType.HOT
    
    print("✅ 异构调度器测试通过")
    return True


def test_cluster_manager():
    """测试集群管理器"""
    print("测试集群管理器...")
    
    from src.cluster.manager import ClusterManager, ClusterConfig, NodeRole, ShardInfo
    
    config = ClusterConfig(
        cluster_key="test-key",
        master_host="127.0.0.1",
        master_port=8000,
        node_name="test-node",
        node_role=NodeRole.MASTER
    )
    
    manager = ClusterManager(config)
    
    # 测试分片注册
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
    
    # 测试状态获取
    status = manager.get_cluster_status()
    assert "is_master" in status
    assert status["is_master"] is True
    
    print("✅ 集群管理器测试通过")
    return True


def test_numa_optimizer():
    """测试NUMA优化器"""
    print("测试NUMA优化器...")
    
    from src.heterogeneous.numa_optimizer import NUMAOptimizer
    
    optimizer = NUMAOptimizer()
    
    # 测试信息获取
    info = optimizer.get_info()
    assert "is_available" in info
    
    print("✅ NUMA优化器测试通过")
    return True


def main():
    """主函数"""
    print("=" * 50)
    print("KCake 基础测试")
    print("=" * 50)
    print()
    
    tests = [
        test_kv_cache,
        test_scheduler,
        test_cluster_manager,
        test_numa_optimizer,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 50)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
