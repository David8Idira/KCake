"""
专家放置器模块

负责专家到设备的实际放置和迁移
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import threading

from .scheduler import ExpertInfo, DeviceInfo, ExpertType, DeviceType

logger = logging.getLogger(__name__)


@dataclass
class PlacementResult:
    """放置结果"""
    expert_id: str
    device_id: str
    success: bool
    time_ms: float
    error: Optional[str] = None


class ExpertPlacer:
    """专家放置器"""
    
    def __init__(self, scheduler: Any, transfer_callback: Optional[Callable] = None):
        """
        初始化专家放置器
        
        Args:
            scheduler: 调度器实例
            transfer_callback: 数据传输回调函数
        """
        self.scheduler = scheduler
        self.transfer_callback = transfer_callback
        
        # 当前放置状态
        self.placements: Dict[str, str] = {}  # expert_id -> device_id
        
        # 放置锁
        self._lock = threading.RLock()
        
        # 统计
        self.stats = {
            "total_placements": 0,
            "successful_placements": 0,
            "failed_placements": 0,
            "total_migration_time_ms": 0,
        }
        
        logger.info("专家放置器初始化完成")
    
    async def place_expert(
        self,
        expert_id: str,
        device_id: str,
        expert_data: bytes
    ) -> PlacementResult:
        """
        放置单个专家
        
        Args:
            expert_id: 专家ID
            device_id: 目标设备ID
            expert_data: 专家数据
            
        Returns:
            PlacementResult: 放置结果
        """
        start_time = time.time()
        
        with self._lock:
            # 检查设备是否存在
            device = self.scheduler.devices.get(device_id)
            if device is None:
                return PlacementResult(
                    expert_id=expert_id,
                    device_id=device_id,
                    success=False,
                    time_ms=0,
                    error=f"设备不存在: {device_id}"
                )
            
            # 检查专家是否存在
            expert = self.scheduler.experts.get(expert_id)
            if expert is None:
                return PlacementResult(
                    expert_id=expert_id,
                    device_id=device_id,
                    success=False,
                    time_ms=0,
                    error=f"专家不存在: {expert_id}"
                )
        
        try:
            logger.info(f"开始放置专家: {expert_id} -> {device_id}")
            
            # 如果专家已在该设备上，直接返回
            if expert.current_device == device_id:
                logger.info(f"专家已在目标设备: {expert_id} @ {device_id}")
                return PlacementResult(
                    expert_id=expert_id,
                    device_id=device_id,
                    success=True,
                    time_ms=0,
                    error=None
                )
            
            # 如果专家在其他设备上，需要迁移
            if expert.current_device is not None:
                # 执行迁移
                success = await self._migrate_expert(
                    expert_id,
                    expert.current_device,
                    device_id,
                    expert_data
                )
                
                if not success:
                    return PlacementResult(
                        expert_id=expert_id,
                        device_id=device_id,
                        success=False,
                        time_ms=time.time() - start_time,
                        error="迁移失败"
                    )
            else:
                # 首次放置，直接复制
                await self._copy_expert(expert_id, device_id, expert_data)
            
            # 更新放置状态
            with self._lock:
                self.placements[expert_id] = device_id
                expert.current_device = device_id
                
                # 更新设备负载
                device.memory_available -= expert.size_bytes
                device.current_load = 1.0 - (device.memory_available / device.memory_total)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # 更新统计
            self.stats["total_placements"] += 1
            self.stats["successful_placements"] += 1
            self.stats["total_migration_time_ms"] += elapsed_ms
            
            logger.info(f"专家放置成功: {expert_id} -> {device_id}, 耗时: {elapsed_ms:.2f}ms")
            
            return PlacementResult(
                expert_id=expert_id,
                device_id=device_id,
                success=True,
                time_ms=elapsed_ms
            )
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            
            self.stats["total_placements"] += 1
            self.stats["failed_placements"] += 1
            
            logger.error(f"专家放置失败: {expert_id} -> {device_id}, 错误: {e}")
            
            return PlacementResult(
                expert_id=expert_id,
                device_id=device_id,
                success=False,
                time_ms=elapsed_ms,
                error=str(e)
            )
    
    async def _migrate_expert(
        self,
        expert_id: str,
        source_device: str,
        target_device: str,
        expert_data: bytes
    ) -> bool:
        """
        迁移专家
        
        Args:
            expert_id: 专家ID
            source_device: 源设备ID
            target_device: 目标设备ID
            expert_data: 专家数据
            
        Returns:
            bool: 是否迁移成功
        """
        logger.info(f"开始迁移专家: {expert_id}, {source_device} -> {target_device}")
        
        try:
            # 从源设备获取数据
            source_data = await self._get_expert_from_device(expert_id, source_device)
            
            if source_data is None:
                logger.error(f"从源设备获取专家数据失败: {expert_id} @ {source_device}")
                return False
            
            # 传输到目标设备
            if self.transfer_callback:
                await self.transfer_callback(expert_id, target_device, source_data)
            else:
                # 直接复制数据
                await self._copy_to_device(expert_id, target_device, source_data)
            
            # 从源设备删除
            await self._remove_expert_from_device(expert_id, source_device)
            
            logger.info(f"专家迁移完成: {expert_id}")
            return True
            
        except Exception as e:
            logger.error(f"专家迁移失败: {e}")
            return False
    
    async def _copy_expert(
        self,
        expert_id: str,
        device_id: str,
        expert_data: bytes
    ) -> bool:
        """复制专家数据到设备"""
        try:
            # 模拟复制操作
            await asyncio.sleep(0.01)
            
            logger.debug(f"专家数据已复制: {expert_id} -> {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"复制专家数据失败: {e}")
            return False
    
    async def _get_expert_from_device(
        self,
        expert_id: str,
        device_id: str
    ) -> Optional[bytes]:
        """从设备获取专家数据"""
        try:
            # 模拟从设备读取
            await asyncio.sleep(0.01)
            
            logger.debug(f"从设备获取专家数据: {expert_id} @ {device_id}")
            return None  # 模拟返回
            
        except Exception as e:
            logger.error(f"从设备获取专家数据失败: {e}")
            return None
    
    async def _copy_to_device(
        self,
        expert_id: str,
        device_id: str,
        data: bytes
    ) -> bool:
        """复制数据到设备"""
        try:
            # 模拟复制操作
            await asyncio.sleep(0.01)
            
            logger.debug(f"数据已复制到设备: {expert_id} -> {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"复制数据到设备失败: {e}")
            return False
    
    async def _remove_expert_from_device(
        self,
        expert_id: str,
        device_id: str
    ) -> bool:
        """从设备删除专家"""
        try:
            # 模拟删除操作
            await asyncio.sleep(0.01)
            
            logger.debug(f"专家已从设备删除: {expert_id} @ {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"从设备删除专家失败: {e}")
            return False
    
    async def place_experts_batch(
        self,
        placements: List[tuple]
    ) -> List[PlacementResult]:
        """
        批量放置专家
        
        Args:
            placements: [(expert_id, device_id, expert_data), ...]
            
        Returns:
            List[PlacementResult]: 放置结果列表
        """
        results = []
        
        for expert_id, device_id, expert_data in placements:
            result = await self.place_expert(expert_id, device_id, expert_data)
            results.append(result)
        
        return results
    
    async def remove_expert(self, expert_id: str) -> bool:
        """
        从设备移除专家
        
        Args:
            expert_id: 专家ID
            
        Returns:
            bool: 是否移除成功
        """
        with self._lock:
            if expert_id not in self.placements:
                logger.warning(f"专家未放置: {expert_id}")
                return False
            
            device_id = self.placements[expert_id]
        
        try:
            # 从设备删除
            success = await self._remove_expert_from_device(expert_id, device_id)
            
            if success:
                with self._lock:
                    del self.placements[expert_id]
                    
                    # 更新专家状态
                    expert = self.scheduler.experts.get(expert_id)
                    if expert:
                        expert.current_device = None
                    
                    # 更新设备状态
                    device = self.scheduler.devices.get(device_id)
                    if device and expert:
                        device.memory_available += expert.size_bytes
                        device.current_load = 1.0 - (device.memory_available / device.memory_total)
                
                logger.info(f"专家已移除: {expert_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"移除专家失败: {e}")
            return False
    
    def get_placement_info(self) -> Dict[str, Any]:
        """获取放置信息"""
        with self._lock:
            return {
                "total_placements": len(self.placements),
                "placements": self.placements.copy(),
                "stats": self.stats.copy(),
            }
    
    def get_expert_device(self, expert_id: str) -> Optional[str]:
        """获取专家当前所在设备"""
        with self._lock:
            return self.placements.get(expert_id)
    
    def get_device_experts(self, device_id: str) -> List[str]:
        """获取设备上的所有专家"""
        with self._lock:
            return [
                expert_id for expert_id, d_id in self.placements.items()
                if d_id == device_id
            ]