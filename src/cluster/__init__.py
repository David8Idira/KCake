"""
KCake Cluster - 分布式集群管理

融合 cake (evilsocket) 的核心优势：
1. mDNS 零配置设备发现
2. Transformer 分片到多设备
3. 权重流式传输与缓存
4. 自动拓扑发现
"""

import asyncio
import logging
import hashlib
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, AsyncIterator, Callable
from enum import Enum
import uuid
import json
import time

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    """节点角色"""
    MASTER = "master"  # 主节点，拥有完整模型
    WORKER = "worker"  # 工作节点，提供算力
    HYBRID = "hybrid"  # 混合节点，既有模型又是 worker


class NodeStatus(Enum):
    """节点状态"""
    CONNECTING = "connecting"
    IDLE = "idle"
    BUSY = "busy"
    DISCONNECTED = "disconnected"


@dataclass
class NodeInfo:
    """集群节点信息"""
    node_id: str
    name: str
    role: NodeRole
    host: str
    port: int
    
    # 设备能力
    device_type: str  # cuda, metal, cpu, vulkan, npu
    memory_total: int  # bytes
    memory_available: int
    compute_score: float  # 相对计算能力 (0-1)
    
    # 状态
    status: NodeStatus = NodeStatus.CONNECTING
    last_heartbeat: float = field(default_factory=time.time)
    
    # 分片信息
    loaded_shards: List[int] = field(default_factory=list)
    
    # 元数据
    tags: Dict[str, str] = field(default_factory=dict)
    version: str = "0.1.0"
    
    @property
    def is_available(self) -> bool:
        return (
            self.status == NodeStatus.IDLE and
            time.time() - self.last_heartbeat < 30
        )
    
    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class ShardInfo:
    """模型分片信息"""
    shard_id: int
    layer_start: int
    layer_end: int
    size_bytes: int
    checksum: str  # CRC32
    owner_node: Optional[str] = None  # master 或 None (需下载)
    cached: bool = False
    cache_path: Optional[str] = None


@dataclass
class ClusterConfig:
    """集群配置"""
    cluster_key: str  # 认证密钥
    master_name: str = "master"
    mdns_enabled: bool = True
    heartbeat_interval: int = 5  # seconds
    connection_timeout: int = 30  # seconds
    max_retries: int = 3
    compression_enabled: bool = True
    compression_level: int = 3  # zstd 压缩级别


class ClusterManager:
    """
    集群管理器
    
    核心功能：
    1. 节点注册与发现 (mDNS)
    2. 分片路由与负载均衡
    3. 权重传输与缓存
    4. 心跳健康检查
    """
    
    def __init__(
        self,
        config: ClusterConfig,
        local_node_name: str,
        local_role: NodeRole,
        local_host: str = "localhost",
        local_port: int = 8000,
    ):
        self.config = config
        self.local_node = NodeInfo(
            node_id=str(uuid.uuid4()),
            name=local_node_name,
            role=local_role,
            host=local_host,
            port=local_port,
            device_type="cpu",
            memory_total=0,
            memory_available=0,
            compute_score=1.0,
        )
        
        # 集群节点
        self.nodes: Dict[str, NodeInfo] = {}
        self.master_node: Optional[NodeInfo] = None
        
        # 分片路由
        self.shard_map: Dict[int, ShardInfo] = {}  # shard_id -> ShardInfo
        self.node_shards: Dict[str, List[int]] = {}  # node_id -> [shard_ids]
        
        # 权重缓存
        self.weight_cache: Dict[str, bytes] = {}  # cache_key -> compressed_weights
        
        # 任务队列
        self.inference_queue: asyncio.Queue = asyncio.Queue()
        
        # 回调
        self.on_node_joined: Optional[Callable] = None
        self.on_node_left: Optional[Callable] = None
        
        # 运行状态
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._discovery_task: Optional[asyncio.Task] = None
        
        logger.info(f"ClusterManager initialized")
        logger.info(f"  Local node: {local_node_name} ({local_role.value})")
        logger.info(f"  Cluster key: {config.cluster_key[:4]}***")
    
    async def start(self) -> None:
        """启动集群管理器"""
        self._running = True
        
        # 启动心跳
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # 启动节点发现
        if self.config.mdns_enabled:
            self._discovery_task = asyncio.create_task(self._mdns_discovery_loop())
        
        # 如果是 master，注册自己
        if self.local_node.role == NodeRole.MASTER:
            await self._register_master()
        
        logger.info("ClusterManager started")
    
    async def stop(self) -> None:
        """停止集群管理器"""
        self._running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        if self._discovery_task:
            self._discovery_task.cancel()
        
        logger.info("ClusterManager stopped")
    
    async def join_cluster(self, master_host: str, master_port: int) -> bool:
        """
        加入集群
        
        Args:
            master_host: 主节点主机
            master_port: 主节点端口
        """
        logger.info(f"Joining cluster at {master_host}:{master_port}")
        
        for attempt in range(self.config.max_retries):
            try:
                # 连接主节点
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(master_host, master_port),
                    timeout=self.config.connection_timeout
                )
                
                # 发送加入请求
                join_msg = {
                    "type": "join",
                    "node": {
                        "node_id": self.local_node.node_id,
                        "name": self.local_node.name,
                        "role": self.local_node.role.value,
                        "device_type": self.local_node.device_type,
                        "memory_total": self.local_node.memory_total,
                        "compute_score": self.local_node.compute_score,
                        "version": self.local_node.version,
                    },
                    "cluster_key": self.config.cluster_key,
                }
                
                writer.write(json.dumps(join_msg).encode())
                await writer.drain()
                
                # 等待确认
                response_data = await asyncio.wait_for(
                    reader.read(4096),
                    timeout=self.config.connection_timeout
                )
                
                response = json.loads(response_data.decode())
                
                if response.get("type") == "join_ack":
                    logger.info(f"Successfully joined cluster")
                    writer.close()
                    await writer.wait_closed()
                    return True
                
            except Exception as e:
                logger.warning(f"Join attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        
        logger.error(f"Failed to join cluster after {self.config.max_retries} attempts")
        return False
    
    async def _register_master(self) -> None:
        """注册主节点"""
        self.nodes[self.local_node.node_id] = self.local_node
        self.master_node = self.local_node
        logger.info("Master node registered")
    
    async def _heartbeat_loop(self) -> None:
        """心跳循环 - 维护节点状态"""
        while self._running:
            try:
                current_time = time.time()
                
                # 检查所有节点心跳
                disconnected = []
                for node_id, node in self.nodes.items():
                    if current_time - node.last_heartbeat > self.config.connection_timeout:
                        if node.status != NodeStatus.DISCONNECTED:
                            node.status = NodeStatus.DISCONNECTED
                            disconnected.append(node_id)
                
                # 通知断开的节点
                for node_id in disconnected:
                    logger.warning(f"Node {self.nodes[node_id].name} disconnected")
                    if self.on_node_left:
                        await self.on_node_left(self.nodes[node_id])
                
                # 如果是 master，广播心跳
                if self.local_node.role in [NodeRole.MASTER, NodeRole.HYBRID]:
                    await self._broadcast_heartbeat()
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _broadcast_heartbeat(self) -> None:
        """广播心跳到所有节点"""
        # 实现心跳广播逻辑
        pass
    
    async def _mdns_discovery_loop(self) -> None:
        """
        mDNS 发现循环
        
        自动发现同一网络中的其他 KCake 节点
        """
        logger.info("mDNS discovery started")
        
        while self._running:
            try:
                # 这里是简化的 mDNS 实现
                # 实际应使用 python-zeroconf 库
                await self._scan_for_nodes()
                await asyncio.sleep(10)  # 每10秒扫描一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"mDNS discovery error: {e}")
    
    async def _scan_for_nodes(self) -> None:
        """扫描可用节点"""
        # TODO: 使用 zeroconf 实现 mDNS 扫描
        # 这是概念实现
        logger.debug("Scanning for KCake nodes...")
    
    async def register_shard(
        self,
        shard_id: int,
        layer_start: int,
        layer_end: int,
        size_bytes: int,
    ) -> None:
        """注册模型分片"""
        self.shard_map[shard_id] = ShardInfo(
            shard_id=shard_id,
            layer_start=layer_start,
            layer_end=layer_end,
            size_bytes=size_bytes,
            checksum="",
            owner_node=self.local_node.node_id,
        )
        
        if self.local_node.node_id not in self.node_shards:
            self.node_shards[self.local_node.node_id] = []
        self.node_shards[self.local_node.node_id].append(shard_id)
        
        logger.info(f"Registered shard {shard_id} (layers {layer_start}-{layer_end})")
    
    async def request_shard(
        self,
        shard_id: int,
        target_node: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        请求获取分片数据
        
        Args:
            shard_id: 分片 ID
            target_node: 指定节点，不指定则从拥有该分片的节点获取
            
        Returns:
            分片数据 (压缩格式)
        """
        if shard_id not in self.shard_map:
            logger.error(f"Shard {shard_id} not found")
            return None
        
        shard_info = self.shard_map[shard_id]
        
        # 生成缓存 key
        cache_key = f"{shard_info.checksum}"
        
        # 检查本地缓存
        if cache_key in self.weight_cache:
            logger.debug(f"Shard {shard_id} found in cache")
            return self.weight_cache[cache_key]
        
        # 从拥有节点获取
        if shard_info.owner_node:
            owner = self.nodes.get(shard_info.owner_node)
            if owner:
                data = await self._fetch_shard_from_node(owner, shard_id)
                if data:
                    # 缓存
                    self.weight_cache[cache_key] = data
                    shard_info.cached = True
                return data
        
        return None
    
    async def _fetch_shard_from_node(
        self,
        node: NodeInfo,
        shard_id: int,
    ) -> Optional[bytes]:
        """从指定节点获取分片"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(node.host, node.port),
                timeout=self.config.connection_timeout
            )
            
            # 发送请求
            request = {
                "type": "get_shard",
                "shard_id": shard_id,
            }
            writer.write(json.dumps(request).encode())
            await writer.drain()
            
            # 接收数据
            data = await asyncio.wait_for(
                reader.read(1024 * 1024 * 100),  # 最大 100MB
                timeout=60
            )
            
            writer.close()
            await writer.wait_closed()
            
            # 校验
            if self.config.compression_enabled:
                data = zlib.decompress(data)
            
            expected_checksum = self.shard_map[shard_id].checksum
            actual_checksum = format(zlib.crc32(data), '08x')
            
            if expected_checksum and actual_checksum != expected_checksum:
                logger.error(f"Shard {shard_id} checksum mismatch")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch shard from {node.name}: {e}")
            return None
    
    async def broadcast_inference_request(
        self,
        request_id: str,
        prompt: str,
        shard_assignments: Dict[int, str],  # shard_id -> node_id
    ) -> AsyncIterator[bytes]:
        """
        广播推理请求到各节点
        
        Yields:
            生成的 tokens
        """
        # 向各节点发送分片任务
        tasks = []
        for shard_id, node_id in shard_assignments.items():
            node = self.nodes.get(node_id)
            if node:
                tasks.append(self._send_shard_task(node, request_id, shard_id, prompt))
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        for result in results:
            if isinstance(result, bytes):
                yield result
    
    async def _send_shard_task(
        self,
        node: NodeInfo,
        request_id: str,
        shard_id: int,
        prompt: str,
    ) -> bytes:
        """发送分片任务到节点"""
        # 实现任务发送
        return b""
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        available_nodes = [n for n in self.nodes.values() if n.is_available]
        
        return {
            "total_nodes": len(self.nodes),
            "available_nodes": len(available_nodes),
            "master": self.master_node.name if self.master_node else None,
            "local_node": self.local_node.name,
            "shards": {
                "total": len(self.shard_map),
                "owned_by_local": len(self.node_shards.get(self.local_node.node_id, [])),
            },
            "cache_size_mb": sum(len(v) for v in self.weight_cache.values()) / (1024*1024),
            "nodes": {
                node_id: {
                    "name": node.name,
                    "role": node.role.value,
                    "status": node.status.value,
                    "device_type": node.device_type,
                    "memory_available_gb": node.memory_available / (1024**3),
                    "compute_score": node.compute_score,
                }
                for node_id, node in self.nodes.items()
            },
        }
    
    async def set_node_status(self, status: NodeStatus) -> None:
        """设置本地节点状态"""
        self.local_node.status = status
        self.local_node.last_heartbeat = time.time()
    
    def get_optimal_shard_distribution(
        self,
        available_nodes: List[NodeInfo],
    ) -> Dict[int, str]:
        """
        获取最优分片分布
        
        根据节点能力分配分片
        """
        distribution = {}
        
        if not available_nodes:
            return distribution
        
        # 按计算能力排序
        sorted_nodes = sorted(
            available_nodes,
            key=lambda n: n.compute_score * n.memory_available,
            reverse=True
        )
        
        # 平均分配分片
        shards_per_node = len(self.shard_map) // len(sorted_nodes)
        
        for i, (shard_id, shard) in enumerate(self.shard_map.items()):
            node_idx = min(i // shards_per_node, len(sorted_nodes) - 1)
            distribution[shard_id] = sorted_nodes[node_idx].node_id
        
        return distribution
