"""
KV缓存管理模块

管理推理过程中的Key-Value缓存，支持内存和磁盘缓存
"""

import logging
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: np.ndarray
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self):
        """更新访问时间"""
        self.last_accessed = time.time()
        self.access_count += 1


class KVCache:
    """KV缓存管理器"""
    
    def __init__(
        self,
        max_size_mb: int = 1024,
        max_entries: int = 10000,
        eviction_policy: str = "lru"
    ):
        """
        初始化KV缓存
        
        Args:
            max_size_mb: 最大缓存大小（MB）
            max_entries: 最大缓存条目数
            eviction_policy: 淘汰策略 (lru, lfu, fifo)
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.eviction_policy = eviction_policy
        
        # 缓存存储
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # 统计信息
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "current_size_bytes": 0,
            "current_entries": 0
        }
        
        logger.info(
            f"KV缓存初始化: max_size={max_size_mb}MB, "
            f"max_entries={max_entries}, policy={eviction_policy}"
        )
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[np.ndarray]: 缓存值，不存在返回None
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats["misses"] += 1
                logger.debug(f"缓存未命中: {key}")
                return None
            
            # 更新访问信息
            entry.touch()
            
            # 根据淘汰策略调整顺序
            if self.eviction_policy == "lru":
                self._cache.move_to_end(key)
            
            self._stats["hits"] += 1
            logger.debug(f"缓存命中: {key}, 访问次数: {entry.access_count}")
            
            return entry.value.copy()
    
    def put(
        self,
        key: str,
        value: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        放入缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            metadata: 元数据
        """
        with self._lock:
            # 计算大小
            size_bytes = value.nbytes
            
            # 检查是否已存在
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats["current_size_bytes"] -= old_entry.size_bytes
                del self._cache[key]
                logger.debug(f"更新缓存: {key}, 大小: {old_entry.size_bytes} -> {size_bytes}")
            
            # 检查是否需要淘汰
            while (
                self._stats["current_size_bytes"] + size_bytes > self.max_size_bytes
                or self._stats["current_entries"] >= self.max_entries
            ) and self._cache:
                self._evict_one()
            
            # 创建新条目
            now = time.time()
            entry = CacheEntry(
                key=key,
                value=value.copy(),
                size_bytes=size_bytes,
                created_at=now,
                last_accessed=now,
                metadata=metadata or {}
            )
            
            self._cache[key] = entry
            self._stats["current_size_bytes"] += size_bytes
            self._stats["current_entries"] += 1
            
            logger.debug(f"缓存已添加: {key}, 大小: {size_bytes} bytes")
    
    def _evict_one(self) -> None:
        """淘汰一个条目"""
        if not self._cache:
            return
        
        # 根据淘汰策略选择要淘汰的条目
        if self.eviction_policy == "lru":
            # LRU: 淘汰最久未使用的
            key = next(iter(self._cache))
        elif self.eviction_policy == "lfu":
            # LFU: 淘汰访问次数最少的
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        else:
            # FIFO: 淘汰最早的
            key = next(iter(self._cache))
        
        entry = self._cache.pop(key)
        self._stats["current_size_bytes"] -= entry.size_bytes
        self._stats["current_entries"] -= 1
        self._stats["evictions"] += 1
        
        logger.debug(f"缓存淘汰: {key}, 淘汰原因: {self.eviction_policy}")
    
    def delete(self, key: str) -> bool:
        """
        删除缓存条目
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否删除成功
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache.pop(key)
            self._stats["current_size_bytes"] -= entry.size_bytes
            self._stats["current_entries"] -= 1
            
            logger.debug(f"缓存已删除: {key}")
            return True
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._stats["current_size_bytes"] = 0
            self._stats["current_entries"] = 0
            
            logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests
                if total_requests > 0
                else 0.0
            )
            
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": hit_rate,
                "current_size_mb": self._stats["current_size_bytes"] / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "current_entries": self._stats["current_entries"],
                "max_entries": self.max_entries,
                "utilization": (
                    self._stats["current_entries"] / self.max_entries
                    if self.max_entries > 0
                    else 0.0
                )
            }
    
    def keys(self) -> List[str]:
        """获取所有缓存键"""
        with self._lock:
            return list(self._cache.keys())
    
    def __len__(self) -> int:
        """获取缓存条目数"""
        with self._lock:
            return self._stats["current_entries"]
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            return key in self._cache


class MultiLevelKVCache:
    """多级KV缓存（内存 + 磁盘）"""
    
    def __init__(
        self,
        memory_cache: Optional[KVCache] = None,
        disk_cache_dir: Optional[str] = None,
        disk_max_size_gb: int = 10
    ):
        """
        初始化多级KV缓存
        
        Args:
            memory_cache: 内存缓存实例
            disk_cache_dir: 磁盘缓存目录
            disk_max_size_gb: 磁盘缓存最大大小（GB）
        """
        self.memory_cache = memory_cache or KVCache(max_size_mb=1024)
        self.disk_cache_dir = disk_cache_dir
        self.disk_max_size_bytes = disk_max_size_gb * 1024 * 1024 * 1024
        
        logger.info(
            f"多级KV缓存初始化: 内存={memory_cache or 'default'}, "
            f"磁盘={disk_cache_dir or 'disabled'}"
        )
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """获取缓存值（先查内存，再查磁盘）"""
        # 先查内存
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # 再查磁盘
        if self.disk_cache_dir:
            return self._get_from_disk(key)
        
        return None
    
    def put(self, key: str, value: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """放入缓存（先放内存，内存满了再放磁盘）"""
        # 先放内存
        self.memory_cache.put(key, value, metadata)
        
        # 如果内存已满，尝试放入磁盘
        if self.disk_cache_dir:
            stats = self.memory_cache.get_stats()
            if stats["utilization"] > 0.9:
                self._put_to_disk(key, value, metadata)
    
    def _get_from_disk(self, key: str) -> Optional[np.ndarray]:
        """从磁盘读取缓存"""
        import os
        disk_path = os.path.join(self.disk_cache_dir, f"{key}.pkl")
        
        if os.path.exists(disk_path):
            try:
                with open(disk_path, "rb") as f:
                    entry = pickle.load(f)
                    return entry["value"]
            except Exception as e:
                logger.warning(f"从磁盘读取缓存失败: {key}, {e}")
        
        return None
    
    def _put_to_disk(self, key: str, value: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """放入磁盘缓存"""
        import os
        
        if not self.disk_cache_dir:
            return
        
        os.makedirs(self.disk_cache_dir, exist_ok=True)
        disk_path = os.path.join(self.disk_cache_dir, f"{key}.pkl")
        
        try:
            entry = {
                "key": key,
                "value": value,
                "metadata": metadata or {},
                "timestamp": time.time()
            }
            with open(disk_path, "wb") as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"写入磁盘缓存失败: {key}, {e}")
    
    def clear(self) -> None:
        """清空所有缓存"""
        self.memory_cache.clear()
        
        if self.disk_cache_dir:
            import shutil
            if os.path.exists(self.disk_cache_dir):
                shutil.rmtree(self.disk_cache_dir)
                os.makedirs(self.disk_cache_dir)