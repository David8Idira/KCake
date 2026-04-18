"""
分片路由器模块

负责推理请求到集群节点的分片路由
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import threading

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """路由结果"""
    shard_id: int
    target_node: str
    layer_range: Tuple[int, int]
    is_local: bool
    estimated_latency_ms: float


class ShardRouter:
    """分片路由器"""
    
    def __init__(self, cluster_manager: Any):
        """
        初始化分片路由器
        
        Args:
            cluster_manager: 集群管理器实例
        """
        self.cluster_manager = cluster_manager
        
        # 分片配置
        self.num_shards = 8  # 默认分片数
        self.layers_per_shard = 0  # 由模型决定
        
        # 路由策略
        self.routing_strategy = "least_load"  # least_load, round_robin, geographic
        
        # 统计
        self.stats = {
            "total_routes": 0,
            "local_routes": 0,
            "remote_routes": 0,
            "failed_routes": 0,
        }
        
        # 锁
        self._lock = threading.RLock()
        
        # 轮询索引
        self._round_robin_index = 0
        
        logger.info("分片路由器初始化完成")
    
    def configure(self, num_shards: int, layers_per_shard: int) -> None:
        """
        配置分片
        
        Args:
            num_shards: 分片数量
            layers_per_shard: 每个分片的层数
        """
        self.num_shards = num_shards
        self.layers_per_shard = layers_per_shard
        
        logger.info(f"分片配置: {num_shards} 个分片, 每分片 {layers_per_shard} 层")
    
    async def route_request(
        self,
        request: Dict[str, Any]
    ) -> List[RouteResult]:
        """
        路由推理请求
        
        Args:
            request: 推理请求
            
        Returns:
            List[RouteResult]: 路由结果列表
        """
        results = []
        
        # 获取请求的模型层信息
        model_name = request.get("model", "")
        num_layers = request.get("num_layers", self.num_shards * self.layers_per_shard)
        
        # 计算需要路由的分片
        shards_per_request = self._calculate_shards(num_layers)
        
        # 获取可用节点
        idle_nodes = self.cluster_manager.get_idle_nodes()
        
        if not idle_nodes:
            logger.warning("没有可用节点")
            return []
        
        # 根据路由策略选择节点
        for i, shard_range in enumerate(shards_per_request):
            shard_id = i
            
            # 路由到这个分片
            result = await self._route_shard(
                shard_id,
                shard_range,
                idle_nodes,
                request
            )
            
            if result:
                results.append(result)
                
                # 更新统计
                with self._lock:
                    self.stats["total_routes"] += 1
                    if result.is_local:
                        self.stats["local_routes"] += 1
                    else:
                        self.stats["remote_routes"] += 1
            else:
                with self._lock:
                    self.stats["failed_routes"] += 1
        
        return results
    
    def _calculate_shards(self, num_layers: int) -> List[Tuple[int, int]]:
        """计算分片范围"""
        shards = []
        
        if self.layers_per_shard > 0:
            # 根据层数计算分片
            num_shards_needed = (num_layers + self.layers_per_shard - 1) // self.layers_per_shard
            
            for i in range(num_shards_needed):
                start_layer = i * self.layers_per_shard
                end_layer = min(start_layer + self.layers_per_shard, num_layers)
                shards.append((start_layer, end_layer))
        else:
            # 平均分片
            layers_per_shard = num_layers // self.num_shards
            remainder = num_layers % self.num_shards
            
            start_layer = 0
            for i in range(self.num_shards):
                end_layer = start_layer + layers_per_shard + (1 if i < remainder else 0)
                shards.append((start_layer, end_layer))
                start_layer = end_layer
        
        return shards
    
    async def _route_shard(
        self,
        shard_id: int,
        layer_range: Tuple[int, int],
        available_nodes: List[Any],
        request: Dict[str, Any]
    ) -> Optional[RouteResult]:
        """
        路由单个分片
        
        Args:
            shard_id: 分片ID
            layer_range: 层范围
            available_nodes: 可用节点列表
            request: 原始请求
            
        Returns:
            Optional[RouteResult]: 路由结果
        """
        if not available_nodes:
            return None
        
        # 根据路由策略选择节点
        target_node = await self._select_node(available_nodes, request)
        
        if target_node is None:
            return None
        
        # 计算估计延迟
        estimated_latency = self._estimate_latency(target_node, request)
        
        return RouteResult(
            shard_id=shard_id,
            target_node=target_node.node_id,
            layer_range=layer_range,
            is_local=(target_node.node_id == self.cluster_manager.self_info.node_id),
            estimated_latency_ms=estimated_latency
        )
    
    async def _select_node(
        self,
        available_nodes: List[Any],
        request: Dict[str, Any]
    ) -> Optional[Any]:
        """
        选择最优节点
        
        Args:
            available_nodes: 可用节点列表
            request: 请求信息
            
        Returns:
            Optional[Any]: 选择的节点
        """
        if not available_nodes:
            return None
        
        if self.routing_strategy == "least_load":
            # 选择负载最低的节点
            return min(
                available_nodes,
                key=lambda n: n.current_load if hasattr(n, 'current_load') else 0
            )
        
        elif self.routing_strategy == "round_robin":
            # 轮询选择
            with self._lock:
                index = self._round_robin_index
                self._round_robin_index = (self._round_robin_index + 1) % len(available_nodes)
            return available_nodes[index]
        
        elif self.routing_strategy == "geographic":
            # 地理位置优化（简化版：随机选择）
            import random
            return random.choice(available_nodes)
        
        else:
            # 默认：选择第一个
            return available_nodes[0]
    
    def _estimate_latency(self, node: Any, request: Dict[str, Any]) -> float:
        """
        估计延迟
        
        Args:
            node: 目标节点
            request: 请求信息
            
        Returns:
            float: 估计延迟（毫秒）
        """
        # 基础延迟
        base_latency = 10  # ms
        
        # 网络延迟（假设本地100ms，远程500ms）
        is_local = node.node_id == self.cluster_manager.self_info.node_id
        network_latency = 10 if is_local else 200
        
        # 负载延迟
        load_latency = (node.current_load if hasattr(node, 'current_load') else 0) * 100
        
        # 内存延迟
        mem_available = node.memory_available if hasattr(node, 'memory_available') else 0
        mem_latency = 50 if mem_available < 1024**3 else 0  # 内存不足增加50ms
        
        return base_latency + network_latency + load_latency + mem_latency
    
    async def distribute_request(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        分布式请求执行
        
        Args:
            request: 推理请求
            
        Returns:
            Dict[str, Any]: 聚合的响应
        """
        # 路由请求
        routes = await self.route_request(request)
        
        if not routes:
            return {"error": "无可用节点"}
        
        # 按节点分组
        node_tasks: Dict[str, List[RouteResult]] = {}
        for route in routes:
            if route.target_node not in node_tasks:
                node_tasks[route.target_node] = []
            node_tasks[route.target_node].append(route)
        
        # 并行执行
        results = await asyncio.gather(
            *[self._execute_on_node(node_id, routes, request) 
              for node_id, routes in node_tasks.items()],
            return_exceptions=True
        )
        
        # 聚合结果
        aggregated = self._aggregate_results(results)
        
        return aggregated
    
    async def _execute_on_node(
        self,
        node_id: str,
        routes: List[RouteResult],
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        在指定节点执行任务
        
        Args:
            node_id: 节点ID
            routes: 该节点的路由结果
            request: 请求信息
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        try:
            # 构建子请求
            sub_request = {
                **request,
                "shard_ids": [r.shard_id for r in routes],
                "layer_ranges": [r.layer_range for r in routes]
            }
            
            # 如果是本节点，直接执行
            if node_id == self.cluster_manager.self_info.node_id:
                return await self._local_execute(sub_request)
            
            # 否则通过RPC调用远程节点
            return await self._remote_execute(node_id, sub_request)
            
        except Exception as e:
            logger.error(f"节点 {node_id} 执行失败: {e}")
            return {"error": str(e), "node_id": node_id}
    
    async def _local_execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """本地执行"""
        # TODO: 调用本地推理引擎
        return {"status": "local_executed", "node": "self"}
    
    async def _remote_execute(self, node_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """远程执行"""
        # TODO: 实现RPC调用
        return {"status": "remote_executed", "node": node_id}
    
    def _aggregate_results(self, results: List[Any]) -> Dict[str, Any]:
        """聚合多个节点的结果"""
        # TODO: 实现结果聚合逻辑
        successful = [r for r in results if isinstance(r, dict) and "error" not in r]
        
        if not successful:
            return {"error": "所有节点执行失败"}
        
        return {
            "status": "aggregated",
            "total_nodes": len(results),
            "successful_nodes": len(successful),
            "results": successful
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        with self._lock:
            total = self.stats["total_routes"]
            
            return {
                "total_routes": total,
                "local_routes": self.stats["local_routes"],
                "remote_routes": self.stats["remote_routes"],
                "failed_routes": self.stats["failed_routes"],
                "local_ratio": self.stats["local_routes"] / total if total > 0 else 0,
                "remote_ratio": self.stats["remote_routes"] / total if total > 0 else 0,
                "success_ratio": (total - self.stats["failed_routes"]) / total if total > 0 else 0,
            }
    
    def set_routing_strategy(self, strategy: str) -> bool:
        """
        设置路由策略
        
        Args:
            strategy: 策略名称
            
        Returns:
            bool: 是否设置成功
        """
        valid_strategies = ["least_load", "round_robin", "geographic"]
        
        if strategy not in valid_strategies:
            logger.error(f"无效的路由策略: {strategy}")
            return False
        
        self.routing_strategy = strategy
        logger.info(f"路由策略已设置为: {strategy}")
        return True