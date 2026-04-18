"""
集群管理器模块

负责任务调度、负载均衡、集群状态维护
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
import threading
import socket
import struct

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    """节点角色"""
    MASTER = "master"
    WORKER = "worker"
    HYBRID = "hybrid"


class NodeStatus(Enum):
    """节点状态"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    FAILING = "failing"


@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    name: str
    role: NodeRole
    host: str
    port: int
    device_type: str
    memory_total: int
    memory_available: int
    compute_score: float
    status: NodeStatus = NodeStatus.IDLE
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShardInfo:
    """分片信息"""
    shard_id: int
    layer_start: int
    layer_end: int
    size_bytes: int
    checksum: str
    owner_node: Optional[str] = None
    cached: bool = False
    access_count: int = 0


@dataclass
class ClusterConfig:
    """集群配置"""
    cluster_key: str
    master_host: str
    master_port: int
    node_name: str
    node_role: NodeRole = NodeRole.WORKER
    heartbeat_interval: int = 10  # 秒
    heartbeat_timeout: int = 30  # 秒
    max_retry: int = 3


class ClusterManager:
    """集群管理器"""
    
    def __init__(
        self,
        config: ClusterConfig,
        on_node_joined: Optional[Callable] = None,
        on_node_left: Optional[Callable] = None,
        on_shard_request: Optional[Callable] = None
    ):
        """
        初始化集群管理器
        
        Args:
            config: 集群配置
            on_node_joined: 节点加入回调
            on_node_left: 节点离开回调
            on_shard_request: 分片请求回调
        """
        self.config = config
        
        # 回调函数
        self.on_node_joined = on_node_joined
        self.on_node_left = on_node_left
        self.on_shard_request = on_shard_request
        
        # 集群状态
        self.is_running = False
        self.is_master = config.node_role in [NodeRole.MASTER, NodeRole.HYBRID]
        
        # 本节点信息
        self.self_info = NodeInfo(
            node_id=str(uuid.uuid4()),
            name=config.node_name,
            role=config.node_role,
            host=self._get_local_ip(),
            port=config.master_port if self.is_master else 0,
            device_type="cpu",  # 后续从设备抽象层获取
            memory_total=0,
            memory_available=0,
            compute_score=1.0,
            status=NodeStatus.IDLE,
            capabilities=["inference", "scheduling"] if self.is_master else ["inference"]
        )
        
        # 集群节点列表
        self.nodes: Dict[str, NodeInfo] = {}
        if self.is_master:
            # Master节点自己也要加入列表
            self.nodes[self.self_info.node_id] = self.self_info
        
        # 分片信息
        self.shards: Dict[int, ShardInfo] = {}
        
        # 锁
        self._lock = threading.RLock()
        
        # 任务
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._server_task: Optional[asyncio.Task] = None
        self._server_socket: Optional[socket.socket] = None
        
        # 统计
        self.stats = {
            "start_time": time.time(),
            "total_requests": 0,
            "failed_requests": 0,
            "nodes_joined": 0,
            "nodes_left": 0,
        }
        
        logger.info(
            f"集群管理器初始化: role={config.node_role.value}, "
            f"master={self.is_master}, name={config.node_name}"
        )
    
    def _get_local_ip(self) -> str:
        """获取本机IP"""
        try:
            # 创建UDP socket连接外部地址来获取本机IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    async def start(self) -> bool:
        """
        启动集群管理器
        
        Returns:
            bool: 是否启动成功
        """
        if self.is_running:
            logger.warning("集群管理器已在运行")
            return True
        
        try:
            self.is_running = True
            
            # 启动心跳任务
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            if self.is_master:
                # Master节点启动服务器
                self._server_task = asyncio.create_task(self._server_loop())
                logger.info(f"Master节点已启动: {self.self_info.host}:{self.self_info.port}")
            else:
                # Worker节点加入集群
                await self._join_cluster()
                logger.info("Worker节点已启动并加入集群")
            
            return True
            
        except Exception as e:
            logger.error(f"启动集群管理器失败: {e}")
            self.is_running = False
            return False
    
    async def stop(self) -> None:
        """停止集群管理器"""
        if not self.is_running:
            return
        
        logger.info("正在停止集群管理器...")
        self.is_running = False
        
        # 取消心跳任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # 关闭服务器
        if self._server_socket:
            self._server_socket.close()
        
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        # 如果是Worker，通知离开集群
        if not self.is_master:
            await self._leave_cluster()
        
        logger.info("集群管理器已停止")
    
    async def _heartbeat_loop(self) -> None:
        """心跳循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                # 检查节点状态
                await self._check_node_health()
                
                # 如果是Master，发送心跳
                if self.is_master:
                    await self._broadcast_heartbeat()
                else:
                    await self._send_heartbeat()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"心跳循环异常: {e}")
    
    async def _check_node_health(self) -> None:
        """检查节点健康状态"""
        with self._lock:
            current_time = time.time()
            offline_nodes = []
            
            for node_id, node in self.nodes.items():
                if node_id == self.self_info.node_id:
                    continue
                
                # 检查心跳超时
                if current_time - node.last_heartbeat > self.config.heartbeat_timeout:
                    offline_nodes.append(node_id)
            
            # 处理离线的节点
            for node_id in offline_nodes:
                node = self.nodes.pop(node_id, None)
                if node:
                    logger.warning(f"节点离线: {node.name} ({node_id})")
                    
                    if self.on_node_left:
                        await self._safe_callback(self.on_node_left, node)
                    
                    self.stats["nodes_left"] += 1
    
    async def _broadcast_heartbeat(self) -> None:
        """广播心跳（Master用）"""
        # 更新自身心跳
        self.self_info.last_heartbeat = time.time()
        
        # TODO: 实现TCP/UDP广播
        # 这里简化处理，实际应该向所有节点发送心跳包
    
    async def _send_heartbeat(self) -> None:
        """发送心跳（Worker用）"""
        try:
            # 构建心跳消息
            heartbeat = {
                "type": "heartbeat",
                "node_id": self.self_info.node_id,
                "name": self.self_info.name,
                "status": self.self_info.status.value,
                "memory_available": self.self_info.memory_available,
                "timestamp": time.time()
            }
            
            # 发送到Master
            # TODO: 实现实际的网络发送
            
        except Exception as e:
            logger.error(f"发送心跳失败: {e}")
    
    async def _server_loop(self) -> None:
        """服务器循环（Master用）"""
        try:
            # 创建TCP服务器
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind(('', self.self_info.port))
            self._server_socket.listen(128)
            
            logger.info(f"集群服务器监听: {self.self_info.port}")
            
            while self.is_running:
                try:
                    client_socket, address = await asyncio.get_event_loop().sock_accept(self._server_socket)
                    asyncio.create_task(self._handle_client(client_socket, address))
                except Exception as e:
                    if self.is_running:
                        logger.error(f"接受连接失败: {e}")
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"服务器循环异常: {e}")
        finally:
            if self._server_socket:
                self._server_socket.close()
    
    async def _handle_client(
        self,
        client_socket: socket.socket,
        address: tuple
    ) -> None:
        """处理客户端连接"""
        try:
            # 接收数据
            data = client_socket.recv(4096)
            if not data:
                return
            
            # 解析消息
            message = json.loads(data.decode())
            msg_type = message.get("type")
            
            if msg_type == "heartbeat":
                await self._handle_heartbeat(message)
            elif msg_type == "join":
                await self._handle_node_join(message, client_socket)
            elif msg_type == "leave":
                await self._handle_node_leave(message)
            elif msg_type == "shard_request":
                await self._handle_shard_request(message)
            else:
                logger.warning(f"未知消息类型: {msg_type}")
                
        except Exception as e:
            logger.error(f"处理客户端请求失败: {e}")
        finally:
            client_socket.close()
    
    async def _handle_heartbeat(self, message: Dict) -> None:
        """处理心跳消息"""
        node_id = message.get("node_id")
        
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].last_heartbeat = message.get("timestamp", time.time())
                self.nodes[node_id].status = NodeStatus(message.get("status", "idle"))
    
    async def _handle_node_join(self, message: Dict, client_socket: socket.socket) -> None:
        """处理节点加入"""
        node_info = NodeInfo(
            node_id=message.get("node_id"),
            name=message.get("name"),
            role=NodeRole(message.get("role", "worker")),
            host=message.get("host"),
            port=message.get("port"),
            device_type=message.get("device_type", "cpu"),
            memory_total=message.get("memory_total", 0),
            memory_available=message.get("memory_available", 0),
            compute_score=message.get("compute_score", 1.0),
            capabilities=message.get("capabilities", [])
        )
        
        with self._lock:
            self.nodes[node_info.node_id] = node_info
        
        # 发送响应
        response = {
            "type": "join_ack",
            "node_id": self.self_info.node_id,
            "cluster_id": self.config.cluster_key,
            "nodes": [n.__dict__ for n in self.nodes.values()]
        }
        client_socket.send(json.dumps(response).encode())
        
        # 调用回调
        if self.on_node_joined:
            await self._safe_callback(self.on_node_joined, node_info)
        
        self.stats["nodes_joined"] += 1
        logger.info(f"节点加入集群: {node_info.name} ({node_info.node_id})")
    
    async def _handle_node_leave(self, message: Dict) -> None:
        """处理节点离开"""
        node_id = message.get("node_id")
        
        with self._lock:
            node = self.nodes.pop(node_id, None)
        
        if node:
            if self.on_node_left:
                await self._safe_callback(self.on_node_left, node)
            
            self.stats["nodes_left"] += 1
            logger.info(f"节点离开集群: {node.name} ({node_id})")
    
    async def _handle_shard_request(self, message: Dict) -> None:
        """处理分片请求"""
        shard_id = message.get("shard_id")
        
        if self.on_shard_request:
            result = await self._safe_callback(
                self.on_shard_request,
                shard_id,
                message.get("node_id")
            )
            
            if result:
                # TODO: 发送分片数据
                pass
    
    async def _join_cluster(self) -> bool:
        """加入集群（Worker用）"""
        retry = 0
        
        while retry < self.config.max_retry:
            try:
                # 连接Master
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.config.master_host, self.config.master_port))
                
                # 发送加入请求
                join_request = {
                    "type": "join",
                    "node_id": self.self_info.node_id,
                    "name": self.self_info.name,
                    "role": self.self_info.role.value,
                    "host": self.self_info.host,
                    "port": self.self_info.port,
                    "device_type": self.self_info.device_type,
                    "memory_total": self.self_info.memory_total,
                    "memory_available": self.self_info.memory_available,
                    "compute_score": self.self_info.compute_score,
                    "capabilities": self.self_info.capabilities
                }
                
                sock.send(json.dumps(join_request).encode())
                
                # 接收响应
                response_data = sock.recv(4096)
                response = json.loads(response_data.decode())
                
                if response.get("type") == "join_ack":
                    # 更新集群节点列表
                    with self._lock:
                        for node_dict in response.get("nodes", []):
                            if node_dict["node_id"] != self.self_info.node_id:
                                node = NodeInfo(**node_dict)
                                self.nodes[node.node_id] = node
                    
                    sock.close()
                    logger.info(f"成功加入集群: master={self.config.master_host}:{self.config.master_port}")
                    return True
                    
            except Exception as e:
                logger.error(f"加入集群失败 (重试 {retry + 1}/{self.config.max_retry}): {e}")
                retry += 1
                await asyncio.sleep(2 ** retry)  # 指数退避
            
            finally:
                try:
                    sock.close()
                except:
                    pass
        
        return False
    
    async def _leave_cluster(self) -> None:
        """离开集群（Worker用）"""
        try:
            # 连接Master发送离开消息
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.config.master_host, self.config.master_port))
            
            leave_message = {
                "type": "leave",
                "node_id": self.self_info.node_id
            }
            
            sock.send(json.dumps(leave_message).encode())
            sock.close()
            
            logger.info("已离开集群")
            
        except Exception as e:
            logger.error(f"离开集群失败: {e}")
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs) -> Any:
        """安全执行回调"""
        try:
            if asyncio.iscoroutinefunction(callback):
                return await callback(*args, **kwargs)
            else:
                return callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"回调执行失败: {e}")
            return None
    
    def register_shard(self, shard_info: ShardInfo) -> bool:
        """
        注册分片
        
        Args:
            shard_info: 分片信息
            
        Returns:
            bool: 是否注册成功
        """
        with self._lock:
            if shard_info.shard_id in self.shards:
                logger.warning(f"分片已存在: {shard_info.shard_id}")
                return False
            
            shard_info.owner_node = self.self_info.node_id
            self.shards[shard_info.shard_id] = shard_info
            
            logger.info(f"分片注册: {shard_info.shard_id} -> {self.self_info.node_id}")
            return True
    
    async def request_shard(self, shard_id: int, from_node: Optional[str] = None) -> Optional[bytes]:
        """
        请求分片数据
        
        Args:
            shard_id: 分片ID
            from_node: 指定节点，None表示任意节点
            
        Returns:
            Optional[bytes]: 分片数据
        """
        with self._lock:
            if shard_id not in self.shards:
                return None
            
            shard = self.shards[shard_id]
            shard.access_count += 1
        
        # TODO: 实现从其他节点获取分片的逻辑
        return None
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        with self._lock:
            return {
                "is_master": self.is_master,
                "is_running": self.is_running,
                "self_node": self.self_info.node_id,
                "self_name": self.self_info.name,
                "total_nodes": len(self.nodes),
                "available_nodes": sum(
                    1 for n in self.nodes.values()
                    if n.status == NodeStatus.IDLE
                ),
                "nodes": {
                    node_id: {
                        "name": node.name,
                        "role": node.role.value,
                        "status": node.status.value,
                        "host": f"{node.host}:{node.port}",
                        "compute_score": node.compute_score,
                        "memory_available_gb": node.memory_available / (1024**3)
                    }
                    for node_id, node in self.nodes.items()
                },
                "total_shards": len(self.shards),
                "uptime_seconds": time.time() - self.stats["start_time"],
                "stats": self.stats.copy()
            }
    
    def get_nodes_by_role(self, role: NodeRole) -> List[NodeInfo]:
        """按角色获取节点列表"""
        with self._lock:
            return [n for n in self.nodes.values() if n.role == role]
    
    def get_idle_nodes(self) -> List[NodeInfo]:
        """获取空闲节点列表"""
        with self._lock:
            return [
                n for n in self.nodes.values()
                if n.status == NodeStatus.IDLE
                and n.node_id != self.self_info.node_id
            ]