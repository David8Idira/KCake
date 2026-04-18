"""
异构调度器核心模块

负责任务调度、负载均衡、专家放置
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class ExpertType(Enum):
    """专家类型"""
    HOT = "hot"      # 高频访问，放置在GPU
    WARM = "warm"    # 中频访问，放置在CPU
    COLD = "cold"    # 低频访问，放置在磁盘/远程


class DeviceType(Enum):
    """设备类型"""
    TPU = "tpu"
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    NPU = "npu"
    CPU = "cpu"
    DISK = "disk"
    REMOTE = "remote"


@dataclass
class ExpertInfo:
    """专家信息"""
    expert_id: str
    name: str
    layer_indices: List[int]
    size_bytes: int
    call_frequency: float = 0.0  # 调用频率 0-1
    expert_type: ExpertType = ExpertType.WARM
    current_device: Optional[str] = None
    last_updated: float = field(default_factory=time.time)


@dataclass
class DeviceInfo:
    """设备信息"""
    device_id: str
    device_type: DeviceType
    name: str
    memory_total: int
    memory_available: int
    compute_score: float  # 0-1 计算能力得分
    bandwidth_score: float  # 0-1 带宽得分
    is_available: bool = True
    current_load: float = 0.0  # 0-1 当前负载


@dataclass
class ScheduleResult:
    """调度结果"""
    expert_id: str
    target_device: str
    migration_needed: bool
    estimated_time_ms: float
    reason: str


class HeteroScheduler:
    """异构调度器"""
    
    # 设备类型优先级映射
    DEVICE_PRIORITY = {
        DeviceType.TPU: 100,
        DeviceType.CUDA: 90,
        DeviceType.METAL: 85,
        DeviceType.VULKAN: 80,
        DeviceType.NPU: 70,
        DeviceType.CPU: 50,
        DeviceType.DISK: 30,
        DeviceType.REMOTE: 10,
    }
    
    # 专家类型到设备类型映射
    EXPERT_DEVICE_MAP = {
        ExpertType.HOT: [DeviceType.TPU, DeviceType.CUDA, DeviceType.METAL],
        ExpertType.WARM: [DeviceType.VULKAN, DeviceType.NPU, DeviceType.CPU],
        ExpertType.COLD: [DeviceType.DISK, DeviceType.REMOTE],
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化异构调度器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 专家和设备注册表
        self.experts: Dict[str, ExpertInfo] = {}
        self.devices: Dict[str, DeviceInfo] = {}
        
        # 调度策略配置
        self.hot_threshold = self.config.get("hot_threshold", 0.8)  # Hot阈值
        self.cold_threshold = self.config.get("cold_threshold", 0.3)  # Cold阈值
        self.load_balance_window = self.config.get("load_balance_window", 100)  # 负载窗口
        
        # 统计信息
        self.stats = {
            "total_schedules": 0,
            "successful_migrations": 0,
            "failed_migrations": 0,
            "cache_hits": 0,
        }
        
        # 锁
        self._lock = threading.RLock()
        
        # 频率统计
        self._frequency_counter: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("异构调度器初始化完成")
    
    def register_device(self, device_info: DeviceInfo) -> bool:
        """
        注册设备
        
        Args:
            device_info: 设备信息
            
        Returns:
            bool: 是否注册成功
        """
        with self._lock:
            if device_info.device_id in self.devices:
                logger.warning(f"设备已注册: {device_info.device_id}")
                return False
            
            self.devices[device_info.device_id] = device_info
            logger.info(
                f"设备注册成功: {device_info.device_id}, "
                f"类型: {device_info.device_type.value}, "
                f"内存: {device_info.memory_available / (1024**3):.2f}GB"
            )
            return True
    
    def unregister_device(self, device_id: str) -> bool:
        """
        注销设备
        
        Args:
            device_id: 设备ID
            
        Returns:
            bool: 是否注销成功
        """
        with self._lock:
            if device_id not in self.devices:
                logger.warning(f"设备不存在: {device_id}")
                return False
            
            del self.devices[device_id]
            logger.info(f"设备已注销: {device_id}")
            return True
    
    def register_expert(self, expert_info: ExpertInfo) -> bool:
        """
        注册专家
        
        Args:
            expert_info: 专家信息
            
        Returns:
            bool: 是否注册成功
        """
        with self._lock:
            if expert_info.expert_id in self.experts:
                logger.warning(f"专家已注册: {expert_info.expert_id}")
                return False
            
            # 自动分类专家类型
            expert_type = self._classify_expert(expert_info.call_frequency)
            expert_info.expert_type = expert_type
            
            self.experts[expert_info.expert_id] = expert_info
            logger.info(
                f"专家注册成功: {expert_info.expert_id}, "
                f"类型: {expert_type.value}, "
                f"大小: {expert_info.size_bytes / (1024**2):.2f}MB"
            )
            return True
    
    def unregister_expert(self, expert_id: str) -> bool:
        """
        注销专家
        
        Args:
            expert_id: 专家ID
            
        Returns:
            bool: 是否注销成功
        """
        with self._lock:
            if expert_id not in self.experts:
                logger.warning(f"专家不存在: {expert_id}")
                return False
            
            del self.experts[expert_id]
            logger.info(f"专家已注销: {expert_id}")
            return True
    
    def update_expert_frequency(self, expert_id: str, frequency: float) -> bool:
        """
        更新专家调用频率
        
        Args:
            expert_id: 专家ID
            frequency: 调用频率 (0-1)
            
        Returns:
            bool: 是否更新成功
        """
        with self._lock:
            if expert_id not in self.experts:
                logger.warning(f"专家不存在: {expert_id}")
                return False
            
            expert = self.experts[expert_id]
            old_type = expert.expert_type
            
            # 更新频率
            expert.call_frequency = frequency
            expert.last_updated = time.time()
            
            # 重新分类
            new_type = self._classify_expert(frequency)
            expert.expert_type = new_type
            
            # 如果类型变化，记录统计
            if old_type != new_type:
                logger.info(
                    f"专家类型变化: {expert_id}, {old_type.value} -> {new_type.value}"
                )
                self._frequency_counter[expert_id].append(frequency)
            
            return True
    
    def _classify_expert(self, frequency: float) -> ExpertType:
        """根据频率分类专家"""
        if frequency >= self.hot_threshold:
            return ExpertType.HOT
        elif frequency <= self.cold_threshold:
            return ExpertType.COLD
        else:
            return ExpertType.WARM
    
    async def schedule_expert(self, expert_id: str) -> Optional[ScheduleResult]:
        """
        调度单个专家
        
        Args:
            expert_id: 专家ID
            
        Returns:
            Optional[ScheduleResult]: 调度结果
        """
        with self._lock:
            if expert_id not in self.experts:
                logger.error(f"专家不存在: {expert_id}")
                return None
            
            expert = self.experts[expert_id]
            
        # 查找最优设备
        target_device = await self._find_optimal_device(expert)
        
        if target_device is None:
            return ScheduleResult(
                expert_id=expert_id,
                target_device="",
                migration_needed=False,
                estimated_time_ms=0,
                reason="无可用设备"
            )
        
        # 计算是否需要迁移
        migration_needed = (
            expert.current_device != target_device.device_id
            and expert.current_device is not None
        )
        
        # 估算迁移时间
        estimated_time_ms = self._estimate_migration_time(
            expert.size_bytes,
            expert.current_device,
            target_device.device_id
        )
        
        result = ScheduleResult(
            expert_id=expert_id,
            target_device=target_device.device_id,
            migration_needed=migration_needed,
            estimated_time_ms=estimated_time_ms,
            reason=f"类型{expert.expert_type.value}, 设备优先级{target_device.device_type.value}"
        )
        
        # 更新统计
        self.stats["total_schedules"] += 1
        
        if migration_needed:
            self.stats["successful_migrations"] += 1
        
        return result
    
    async def schedule_experts(
        self,
        expert_ids: Optional[List[str]] = None
    ) -> List[ScheduleResult]:
        """
        批量调度专家
        
        Args:
            expert_ids: 专家ID列表，None表示调度所有
            
        Returns:
            List[ScheduleResult]: 调度结果列表
        """
        if expert_ids is None:
            expert_ids = list(self.experts.keys())
        
        results = []
        for expert_id in expert_ids:
            result = await self.schedule_expert(expert_id)
            if result:
                results.append(result)
        
        return results
    
    async def _find_optimal_device(
        self,
        expert: ExpertInfo
    ) -> Optional[DeviceInfo]:
        """查找最优设备"""
        # 获取该专家类型支持的设备类型
        target_types = self.EXPERT_DEVICE_MAP.get(expert.expert_type, [DeviceType.CPU])
        
        # 过滤可用设备
        candidates = [
            d for d in self.devices.values()
            if d.is_available
            and d.device_type in target_types
            and d.memory_available >= expert.size_bytes
            and d.current_load < 0.9  # 负载小于90%
        ]
        
        if not candidates:
            # 降级：尝试所有可用设备
            candidates = [
                d for d in self.devices.values()
                if d.is_available
                and d.memory_available >= expert.size_bytes
                and d.current_load < 0.9
            ]
        
        if not candidates:
            logger.warning(f"找不到可用设备: {expert.expert_id}")
            return None
        
        # 计算每个设备的得分
        def device_score(device: DeviceInfo) -> float:
            # 基础优先级得分
            base_score = self.DEVICE_PRIORITY.get(device.device_type, 0)
            
            # 负载调整（负载越低得分越高）
            load_score = (1.0 - device.current_load) * 20
            
            # 可用内存调整
            mem_ratio = device.memory_available / device.memory_total
            mem_score = mem_ratio * 10
            
            # 当前专家已经在该设备上，得分提高
            if device.device_id == expert.current_device:
                return base_score + load_score + mem_score + 50
            
            return base_score + load_score + mem_score
        
        # 选择得分最高的设备
        optimal_device = max(candidates, key=device_score)
        
        return optimal_device
    
    def _estimate_migration_time(
        self,
        size_bytes: int,
        source_device: Optional[str],
        target_device: str
    ) -> float:
        """估算迁移时间（毫秒）"""
        if source_device == target_device or source_device is None:
            return 0.0
        
        # 获取目标设备带宽
        target = self.devices.get(target_device)
        if target is None:
            return float('inf')
        
        # 估算带宽 (GB/s)
        bandwidth = target.bandwidth_score * 50  # 假设最大50GB/s
        
        # 计算迁移时间
        size_gb = size_bytes / (1024 ** 3)
        time_s = size_gb / bandwidth
        
        return time_s * 1000  # 转换为毫秒
    
    def get_expert_placement(self) -> Dict[str, str]:
        """获取当前专家放置情况"""
        with self._lock:
            return {
                expert_id: expert.current_device or "not_placed"
                for expert_id, expert in self.experts.items()
            }
    
    def get_device_utilization(self) -> Dict[str, Dict[str, Any]]:
        """获取设备利用率"""
        with self._lock:
            utilization = {}
            for device_id, device in self.devices.items():
                # 计算该设备上的专家数量
                experts_on_device = [
                    e for e in self.experts.values()
                    if e.current_device == device_id
                ]
                
                total_size = sum(e.size_bytes for e in experts_on_device)
                
                utilization[device_id] = {
                    "device_type": device.device_type.value,
                    "load": device.current_load,
                    "memory_used_bytes": total_size,
                    "memory_total_bytes": device.memory_total,
                    "memory_available_bytes": device.memory_available,
                    "experts_count": len(experts_on_device),
                }
            
            return utilization
    
    def get_stats(self) -> Dict[str, Any]:
        """获取调度统计"""
        return self.stats.copy()
    
    def balance_load(self) -> List[ScheduleResult]:
        """负载均衡"""
        results = []
        
        with self._lock:
            # 找出高负载设备
            high_load_devices = [
                d for d in self.devices.values()
                if d.current_load > 0.8
            ]
            
            # 找出低负载设备
            low_load_devices = [
                d for d in self.devices.values()
                if d.current_load < 0.3
            ]
            
            if not high_load_devices or not low_load_devices:
                logger.info("负载已均衡，无需调整")
                return results
            
            # 从高负载设备迁移专家到低负载设备
            for high_device in high_load_devices:
                experts_to_migrate = [
                    e for e in self.experts.values()
                    if e.current_device == high_device.device_id
                    and e.expert_type != ExpertType.HOT  # 不迁移Hot专家
                ]
                
                for expert in experts_to_migrate[:3]:  # 最多迁移3个
                    result = ScheduleResult(
                        expert_id=expert.expert_id,
                        target_device=low_load_devices[0].device_id,
                        migration_needed=True,
                        estimated_time_ms=self._estimate_migration_time(
                            expert.size_bytes,
                            expert.current_device,
                            low_load_devices[0].device_id
                        ),
                        reason="负载均衡"
                    )
                    results.append(result)
        
        return results