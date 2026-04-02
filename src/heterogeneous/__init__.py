"""
KCake Heterogeneous Scheduler - 异构调度器

融合 ktransformers 的核心优势：
1. CPU-GPU 专家调度
2. NUMA-aware 内存管理
3. AMX/AVX 硬件加速
4. MoE 优化
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ExpertType(Enum):
    """专家类型"""
    HOT = "hot"  # 频繁调用的专家 -> GPU
    WARM = "warm"  # 中等频率 -> CPU
    COLD = "cold"  # 很少调用 -> Disk/Remote


@dataclass
class Expert:
    """MoE 专家"""
    expert_id: int
    name: str
    parameters: int  # 参数量
    call_frequency: float = 0.0
    last_call_time: float = 0.0
    expert_type: ExpertType = ExpertType.COLD
    device_id: str = "cpu"
    
    # 性能指标
    forward_time_ms: float = 0.0
    memory_usage_bytes: int = 0


@dataclass
class PlacementPolicy:
    """放置策略配置"""
    hot_threshold: float = 0.8  # 频率 > 0.8 -> GPU
    warm_threshold: float = 0.3  # 0.3 < 频率 < 0.8 -> CPU
    cold_threshold: float = 0.3  # 频率 < 0.3 -> Disk
    gpu_memory_ratio: float = 0.9  # GPU 内存使用上限
    cpu_memory_ratio: float = 0.7  # CPU 内存使用上限
    enable_prefix_cache: bool = True
    enable_expert_batching: bool = True


@dataclass
class DeviceMemory:
    """设备内存状态"""
    device_id: str
    device_type: str
    total_bytes: int
    used_bytes: int
    available_bytes: int
    
    @property
    def usage_ratio(self) -> float:
        return self.used_bytes / self.total_bytes if self.total_bytes > 0 else 0
    
    @property
    def is_available(self) -> bool:
        return self.available_bytes > 0


class HeteroScheduler:
    """
    异构调度器
    
    核心功能：
    1. 专家放置：根据调用频率和设备能力决定专家放置位置
    2. 动态迁移：运行时根据统计信息调整专家位置
    3. NUMA 感知：为多插槽服务器优化内存分配
    4. 负载均衡：在多个设备间均衡分配计算任务
    """
    
    def __init__(
        self,
        placement_policy: Optional[PlacementPolicy] = None,
        enable_dynamic_migration: bool = True,
        numa_aware: bool = True,
    ):
        self.placement_policy = placement_policy or PlacementPolicy()
        self.enable_dynamic_migration = enable_dynamic_migration
        self.numa_aware = numa_aware
        
        self.experts: Dict[int, Expert] = {}
        self.device_memories: Dict[str, DeviceMemory] = {}
        self.numa_nodes: Dict[int, List[str]] = {}  # NUMA node -> device_ids
        
        # 调度统计
        self.scheduling_stats = {
            "total_schedules": 0,
            "gpu_schedules": 0,
            "cpu_schedules": 0,
            "remote_schedules": 0,
            "expert_migrations": 0,
        }
        
        logger.info("HeteroScheduler initialized")
        logger.info(f"  NUMA-aware: {numa_aware}")
        logger.info(f"  Dynamic migration: {enable_dynamic_migration}")
    
    def register_device(
        self,
        device_id: str,
        device_type: str,
        total_memory: int,
        numa_node: Optional[int] = None,
        compute_units: int = 1,
    ) -> None:
        """注册设备"""
        self.device_memories[device_id] = DeviceMemory(
            device_id=device_id,
            device_type=device_type,
            total_bytes=total_memory,
            used_bytes=0,
            available_bytes=total_memory,
        )
        
        if numa_node is not None and self.numa_aware:
            if numa_node not in self.numa_nodes:
                self.numa_nodes[numa_node] = []
            self.numa_nodes[numa_node].append(device_id)
        
        logger.info(f"Registered device: {device_id} ({device_type}), "
                   f"Memory: {total_memory / (1024**3):.2f} GB, "
                   f"NUMA: {numa_node}")
    
    def register_moe_experts(
        self,
        expert_ids: List[int],
        expert_names: List[str],
        parameter_counts: List[int],
    ) -> None:
        """注册 MoE 专家"""
        for eid, name, params in zip(expert_ids, expert_names, parameter_counts):
            self.experts[eid] = Expert(
                expert_id=eid,
                name=name,
                parameters=params,
            )
        
        logger.info(f"Registered {len(expert_ids)} MoE experts")
    
    def update_expert_frequency(
        self,
        expert_id: int,
        call_frequency: float,
        forward_time_ms: float,
    ) -> None:
        """更新专家调用统计"""
        if expert_id not in self.experts:
            logger.warning(f"Expert {expert_id} not found")
            return
        
        expert = self.experts[expert_id]
        expert.call_frequency = call_frequency
        expert.forward_time_ms = forward_time_ms
        expert.last_call_time = asyncio.get_event_loop().time()
        
        # 动态迁移检查
        if self.enable_dynamic_migration:
            asyncio.create_task(self._check_and_migrate(expert_id))
    
    async def _check_and_migrate(self, expert_id: int) -> None:
        """检查是否需要迁移专家"""
        expert = self.experts[expert_id]
        new_type = self._classify_expert(expert.call_frequency)
        
        if new_type != expert.expert_type:
            logger.info(f"Expert {expert_id} migrating: {expert.expert_type.value} -> {new_type.value}")
            await self._migrate_expert(expert_id, new_type)
            self.scheduling_stats["expert_migrations"] += 1
    
    def _classify_expert(self, frequency: float) -> ExpertType:
        """根据调用频率分类专家"""
        policy = self.placement_policy
        if frequency >= policy.hot_threshold:
            return ExpertType.HOT
        elif frequency >= policy.warm_threshold:
            return ExpertType.WARM
        else:
            return ExpertType.COLD
    
    async def _migrate_expert(
        self,
        expert_id: int,
        new_type: ExpertType,
    ) -> None:
        """执行专家迁移"""
        expert = self.experts[expert_id]
        old_device = expert.device_id
        
        # 选择目标设备
        target_device = self._select_device_for_expert(new_type, expert)
        
        if target_device and target_device != old_device:
            logger.info(f"Migrating expert {expert_id} from {old_device} to {target_device}")
            expert.device_id = target_device
            expert.expert_type = new_type
            
            # 更新内存状态
            self._update_memory_usage(old_device, -expert.memory_usage_bytes)
            self._update_memory_usage(target_device, expert.memory_usage_bytes)
    
    def _select_device_for_expert(
        self,
        expert_type: ExpertType,
        expert: Expert,
    ) -> Optional[str]:
        """为专家选择最佳设备"""
        candidates = []
        
        for device_id, memory in self.device_memories.items():
            if not memory.is_available:
                continue
            
            # 检查是否有足够内存
            if expert.memory_usage_bytes > memory.available_bytes:
                continue
            
            # 计算设备得分
            score = self._calculate_device_score(device_id, expert_type, expert)
            candidates.append((device_id, score))
        
        if not candidates:
            return None
        
        # 选择得分最高的设备
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _calculate_device_score(
        self,
        device_id: str,
        expert_type: ExpertType,
        expert: Expert,
    ) -> float:
        """计算设备得分"""
        memory = self.device_memories[device_id]
        
        # 基础分数：内存可用率
        memory_score = memory.usage_ratio
        
        # 设备类型匹配
        device_type = memory.device_type
        
        if expert_type == ExpertType.HOT:
            # GPU 优先
            if device_type in ["cuda", "metal", "vulkan"]:
                return 1.0 - memory_score
        elif expert_type == ExpertType.WARM:
            # CPU 可接受
            if device_type == "cpu":
                return 0.7 - memory_score * 0.3
        else:  # COLD
            # 远程/磁盘优先
            if device_type in ["remote", "disk"]:
                return 0.9 - memory_score * 0.1
        
        return 0.3 - memory_score * 0.3
    
    def _update_memory_usage(self, device_id: str, delta_bytes: int) -> None:
        """更新设备内存使用"""
        if device_id in self.device_memories:
            memory = self.device_memories[device_id]
            memory.used_bytes = max(0, memory.used_bytes + delta_bytes)
            memory.available_bytes = memory.total_bytes - memory.used_bytes
    
    async def schedule_experts(
        self,
        expert_ids: List[int],
        available_devices: List[str],
    ) -> Dict[int, str]:
        """
        调度专家到设备
        
        Returns:
            expert_id -> device_id 映射
        """
        self.scheduling_stats["total_schedules"] += 1
        
        schedule = {}
        remaining_experts = expert_ids.copy()
        remaining_devices = available_devices.copy()
        
        # 第一阶段：按频率排序，高频优先选择
        sorted_experts = sorted(
            [(eid, self.experts.get(eid)) for eid in expert_ids],
            key=lambda x: x[1].call_frequency if x[1] else 0,
            reverse=True
        )
        
        for expert_id, expert in sorted_experts:
            if expert is None:
                continue
            
            expert_type = self._classify_expert(expert.call_frequency)
            
            # 选择目标设备
            target_device = None
            for device_id in remaining_devices:
                memory = self.device_memories.get(device_id, None)
                if memory and memory.available_bytes >= expert.memory_usage_bytes:
                    if expert_type == ExpertType.HOT and memory.device_type in ["cuda", "metal"]:
                        target_device = device_id
                        break
                    elif expert_type == ExpertType.WARM and memory.device_type == "cpu":
                        target_device = device_id
                        break
            
            if target_device is None:
                # 回退到任何可用设备
                for device_id in remaining_devices:
                    memory = self.device_memories.get(device_id, None)
                    if memory and memory.available_bytes >= expert.memory_usage_bytes:
                        target_device = device_id
                        break
            
            if target_device:
                schedule[expert_id] = target_device
                remaining_devices.remove(target_device)
                self._update_memory_usage(target_device, expert.memory_usage_bytes)
                
                if memory.device_type in ["cuda", "metal", "vulkan"]:
                    self.scheduling_stats["gpu_schedules"] += 1
                elif memory.device_type == "cpu":
                    self.scheduling_stats["cpu_schedules"] += 1
                else:
                    self.scheduling_stats["remote_schedules"] += 1
        
        return schedule
    
    def get_numa_optimal_mapping(
        self,
        expert_ids: List[int],
    ) -> Dict[int, int]:
        """
        获取 NUMA 最优的专家-设备映射
        
        对于多插槽服务器，确保跨 NUMA 节点的访问最小化
        """
        numa_mapping = {}
        
        for numa_node, device_ids in self.numa_nodes.items():
            # 获取该 NUMA 节点上的设备
            local_devices = device_ids
            
            # 该节点上的专家数量
            experts_per_node = len(expert_ids) // len(self.numa_nodes)
            start_idx = list(self.numa_nodes.keys()).index(numa_node) * experts_per_node
            end_idx = start_idx + experts_per_node
            
            node_experts = expert_ids[start_idx:end_idx]
            
            # 优先分配到本地设备
            for i, expert_id in enumerate(node_experts):
                if i < len(local_devices):
                    numa_mapping[expert_id] = local_devices[i]
        
        return numa_mapping
    
    def get_stats(self) -> Dict[str, Any]:
        """获取调度统计"""
        return {
            **self.scheduling_stats,
            "device_memory": {
                did: {
                    "type": mem.device_type,
                    "total_gb": mem.total_bytes / (1024**3),
                    "used_gb": mem.used_bytes / (1024**3),
                    "available_gb": mem.available_bytes / (1024**3),
                    "usage_ratio": f"{mem.usage_ratio:.1%}",
                }
                for did, mem in self.device_memories.items()
            },
            "expert_types": {
                etype.value: sum(
                    1 for e in self.experts.values() if e.expert_type == etype
                )
                for etype in ExpertType
            },
        }
    
    async def prefetch_experts(
        self,
        expert_ids: List[int],
        target_device: str,
    ) -> None:
        """预取专家到指定设备"""
        logger.info(f"Prefetching experts {expert_ids} to {target_device}")
        # 实际实现会触发模型权重加载
        await asyncio.sleep(0.1)  # 模拟预取
    
    def enable_prefix_cache(
        self,
        cache_size: int = 1024,
    ) -> None:
        """启用前缀缓存 - 3层 (GPU-CPU-Disk)"""
        self.placement_policy.enable_prefix_cache = True
        logger.info(f"Prefix cache enabled, size: {cache_size}")
    
    async def optimize_batch_expert_calls(
        self,
        expert_call_sequence: List[Tuple[int, int]],
    ) -> List[List[int]]:
        """
        批量优化专家调用
        
        将可以合并的专家调用合并以提高效率
        """
        batches = []
        current_batch = []
        
        for expert_id, call_count in expert_call_sequence:
            if not current_batch:
                current_batch.append(expert_id)
            elif expert_id == current_batch[-1]:
                # 连续调用同一专家
                current_batch.append(expert_id)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [expert_id]
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
