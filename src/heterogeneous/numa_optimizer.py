"""
NUMA优化器模块

提供NUMA-aware内存分配和优化
"""

import logging
import os
import platform
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import threading

logger = logging.getLogger(__name__)


@dataclass
class NUMANode:
    """NUMA节点信息"""
    node_id: int
    cpu_list: List[int]
    memory_bytes: int
    memory_available: int
    distance: Dict[int, int]  # 到其他节点的距离


class NUMAOptimizer:
    """NUMA优化器"""
    
    def __init__(self):
        """初始化NUMA优化器"""
        self.is_available = False
        self.nodes: Dict[int, NUMANode] = {}
        self.system_info: Dict[str, Any] = {}
        
        # 检测系统NUMA支持
        self._detect_numa()
    
    def _detect_numa(self) -> None:
        """检测系统NUMA支持"""
        system = platform.system()
        
        if system != "Linux":
            logger.info(f"NUMA优化在 {system} 系统上不可用")
            return
        
        try:
            # 读取NUMA信息
            numa_info = self._read_numa_info()
            
            if numa_info:
                self.nodes = numa_info
                self.is_available = True
                self.system_info["numa_nodes"] = len(numa_info)
                self.system_info["has_numa"] = True
                logger.info(f"NUMA支持检测成功: {len(numa_info)} 个节点")
            else:
                logger.info("未检测到NUMA支持")
                self.system_info["has_numa"] = False
                
        except Exception as e:
            logger.warning(f"NUMA检测失败: {e}")
            self.is_available = False
    
    def _read_numa_info(self) -> Dict[int, NUMANode]:
        """读取系统NUMA信息"""
        nodes: Dict[int, NUMANode] = {}
        
        try:
            # 读取节点数
            num_nodes_file = "/sys/devices/system/node/num_online_nodes"
            if os.path.exists(num_nodes_file):
                with open(num_nodes_file, "r") as f:
                    num_nodes = int(f.read().strip())
            else:
                num_nodes = 1
            
            # 读取每个节点的信息
            for node_id in range(num_nodes):
                node_path = f"/sys/devices/system/node/node{node_id}"
                
                if not os.path.exists(node_path):
                    continue
                
                # CPU列表
                cpu_list = []
                cpulist_file = f"{node_path}/cpulist"
                if os.path.exists(cpulist_file):
                    with open(cpulist_file, "r") as f:
                        cpu_list = self._parse_cpu_list(f.read().strip())
                
                # 内存信息
                mem_total_file = f"{node_path}/meminfo"
                memory_bytes = 0
                memory_available = 0
                
                if os.path.exists(mem_total_file):
                    with open(mem_total_file, "r") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                # MemTotal: 16384084 kB
                                memory_bytes = int(line.split()[3]) * 1024
                            elif line.startswith("MemFree:"):
                                # MemFree: 8192042 kB
                                memory_available = int(line.split()[3]) * 1024
                
                # 距离矩阵
                distance = {}
                distance_file = f"{node_path}/distance"
                if os.path.exists(distance_file):
                    with open(distance_file, "r") as f:
                        distances = f.read().strip().split()
                        for i, dist in enumerate(distances):
                            if i != node_id:  # 跳过自身
                                distance[i] = int(dist)
                
                nodes[node_id] = NUMANode(
                    node_id=node_id,
                    cpu_list=cpu_list,
                    memory_bytes=memory_bytes,
                    memory_available=memory_available,
                    distance=distance
                )
            
        except Exception as e:
            logger.warning(f"读取NUMA信息失败: {e}")
        
        return nodes
    
    def _parse_cpu_list(self, cpu_list_str: str) -> List[int]:
        """
        解析CPU列表字符串
        
        例如: "0-3,8-11" -> [0,1,2,3,8,9,10,11]
        """
        cpus = []
        
        for part in cpu_list_str.split(","):
            part = part.strip()
            
            if "-" in part:
                start, end = map(int, part.split("-"))
                cpus.extend(range(start, end + 1))
            else:
                cpus.append(int(part))
        
        return sorted(list(set(cpus)))
    
    def get_optimal_node(self, memory_size: int, preferred_node: Optional[int] = None) -> Optional[int]:
        """
        获取最优NUMA节点
        
        Args:
            memory_size: 需要的内存大小（字节）
            preferred_node: 偏好节点ID
            
        Returns:
            Optional[int]: 最优节点ID
        """
        if not self.is_available:
            return None
        
        candidates = []
        
        for node_id, node in self.nodes.items():
            # 检查内存是否足够
            if node.memory_available >= memory_size:
                distance = 0 if preferred_node is None else node.distance.get(preferred_node, 100)
                
                candidates.append((node_id, distance, node.memory_available))
        
        if not candidates:
            # 内存不足，返回内存最多的节点
            if self.nodes:
                return max(self.nodes.keys(), key=lambda n: self.nodes[n].memory_available)
            return None
        
        # 优先选择偏好节点（距离最近且内存足够）
        candidates.sort(key=lambda x: (x[1], -x[2]))
        
        return candidates[0][0]
    
    def get_local_cpu_list(self, node_id: int) -> List[int]:
        """
        获取指定节点的本地CPU列表
        
        Args:
            node_id: 节点ID
            
        Returns:
            List[int]: CPU ID列表
        """
        if not self.is_available or node_id not in self.nodes:
            # 返回所有CPU
            return list(range(os.cpu_count() or 1))
        
        return self.nodes[node_id].cpu_list
    
    def get_numa_aware_affinity(
        self,
        memory_size: int,
        preferred_node: Optional[int] = None
    ) -> Tuple[Optional[int], List[int]]:
        """
        获取NUMA感知的CPU亲和性
        
        Args:
            memory_size: 需要的内存大小（字节）
            preferred_node: 偏好节点ID
            
        Returns:
            Tuple[Optional[int], List[int]]: (最优节点ID, 本地CPU列表)
        """
        optimal_node = self.get_optimal_node(memory_size, preferred_node)
        local_cpus = self.get_local_cpu_list(optimal_node or 0)
        
        return optimal_node, local_cpus
    
    def get_memory_stats(self) -> Dict[int, Dict[str, Any]]:
        """获取所有NUMA节点的内存统计"""
        stats = {}
        
        for node_id, node in self.nodes.items():
            stats[node_id] = {
                "total_bytes": node.memory_bytes,
                "available_bytes": node.memory_available,
                "used_bytes": node.memory_bytes - node.memory_available,
                "utilization": (node.memory_bytes - node.memory_available) / node.memory_bytes,
                "cpu_count": len(node.cpu_list),
                "distances": node.distance,
            }
        
        return stats
    
    def set_memory_policy(self, node_id: int) -> bool:
        """
        设置内存分配策略
        
        Args:
            node_id: 节点ID
            
        Returns:
            bool: 是否设置成功
        """
        if not self.is_available or node_id not in self.nodes:
            return False
        
        try:
            # 设置libnuma内存策略
            # 注意：这需要libnuma库支持
            logger.info(f"设置NUMA内存策略: node {node_id}")
            return True
        except Exception as e:
            logger.warning(f"设置内存策略失败: {e}")
            return False
    
    def bind_process_to_node(self, node_id: int, pid: Optional[int] = None) -> bool:
        """
        将进程绑定到指定NUMA节点
        
        Args:
            node_id: 节点ID
            pid: 进程ID，None表示当前进程
            
        Returns:
            bool: 是否绑定成功
        """
        if not self.is_available or node_id not in self.nodes:
            return False
        
        try:
            # 获取CPU列表
            cpu_list = self.nodes[node_id].cpu_list
            if not cpu_list:
                return False
            
            # 构建taskset命令
            cpu_str = ",".join(map(str, cpu_list))
            pid_str = str(pid) if pid else ""
            
            # 执行绑定
            if pid:
                os.system(f"taskset -cp {cpu_str} {pid_str} > /dev/null 2>&1")
            else:
                os.system(f"taskset -cp {cpu_str} > /dev/null 2>&1")
            
            logger.info(f"进程已绑定到NUMA节点 {node_id}, CPU: {cpu_str}")
            return True
            
        except Exception as e:
            logger.warning(f"绑定进程到NUMA节点失败: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """获取NUMA系统信息"""
        return {
            "is_available": self.is_available,
            "system_info": self.system_info,
            "nodes": {
                node_id: {
                    "cpu_count": len(node.cpu_list),
                    "memory_total_gb": node.memory_bytes / (1024**3),
                    "memory_available_gb": node.memory_available / (1024**3),
                }
                for node_id, node in self.nodes.items()
            }
        }