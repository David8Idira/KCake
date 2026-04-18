"""
节点发现模块

提供mDNS零配置网络发现
"""

import asyncio
import logging
import socket
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
import json

logger = logging.getLogger(__name__)


class NodeDiscovery(ABC):
    """节点发现基类"""
    
    @abstractmethod
    async def start(self) -> None:
        """启动发现服务"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """停止发现服务"""
        pass
    
    @abstractmethod
    async def discover_nodes(self) -> List[Dict[str, Any]]:
        """发现节点"""
        pass
    
    @abstractmethod
    async def announce(self, node_info: Dict[str, Any]) -> None:
        """广播节点信息"""
        pass


class mDNSDiscovery(NodeDiscovery):
    """mDNS节点发现实现"""
    
    SERVICE_TYPE = "_kcake._tcp"
    SERVICE_NAME = "KCake Cluster"
    
    def __init__(
        self,
        port: int = 5353,
        domain: str = "local"
    ):
        """
        初始化mDNS发现
        
        Args:
            port: mDNS端口
            domain: mDNS域名
        """
        self.port = port
        self.domain = domain
        self.is_running = False
        
        # 发现的节点缓存
        self.discovered_nodes: Dict[str, Dict[str, Any]] = {}
        
        # 回调
        self.on_node_discovered: Optional[Callable] = None
        self.on_node_expired: Optional[Callable] = None
        
        # 锁
        self._lock = threading.RLock()
        
        # socket
        self._socket: Optional[socket.socket] = None
        
        logger.info(f"mDNS发现初始化: port={port}, domain={domain}")
    
    async def start(self) -> None:
        """启动mDNS服务"""
        if self.is_running:
            return
        
        try:
            # 创建UDP socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            try:
                self._socket.bind(('0.0.0.0', self.port))
            except OSError:
                self._socket.bind(('', self.port))
            
            self._socket.setblocking(False)
            
            self.is_running = True
            
            # 启动监听任务
            asyncio.create_task(self._listen_loop())
            
            logger.info(f"mDNS服务已启动: port={self.port}")
            
        except Exception as e:
            logger.error(f"启动mDNS服务失败: {e}")
            raise
    
    async def stop(self) -> None:
        """停止mDNS服务"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self._socket:
            self._socket.close()
            self._socket = None
        
        logger.info("mDNS服务已停止")
    
    async def _listen_loop(self) -> None:
        """监听循环"""
        loop = asyncio.get_event_loop()
        
        while self.is_running:
            try:
                data, addr = await loop.sock_recvfrom(self._socket, 4096)
                
                # 解析mDNS响应
                await self._handle_mdns_response(data, addr)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.is_running:
                    logger.debug(f"mDNS监听异常: {e}")
    
    async def _handle_mdns_response(
        self,
        data: bytes,
        addr: tuple
    ) -> None:
        """处理mDNS响应"""
        try:
            # 简化处理：直接解析JSON
            # 实际mDNS需要解析DNS记录
            message = json.loads(data.decode())
            
            if message.get("type") == "node_announce":
                node_info = message.get("node")
                if node_info:
                    await self._add_discovered_node(node_info)
                    
        except Exception as e:
            logger.debug(f"处理mDNS响应失败: {e}")
    
    async def _add_discovered_node(self, node_info: Dict[str, Any]) -> None:
        """添加发现的节点"""
        node_id = node_info.get("node_id")
        
        with self._lock:
            is_new = node_id not in self.discovered_nodes
            self.discovered_nodes[node_id] = node_info
        
        if is_new:
            logger.info(f"发现新节点: {node_info.get('name')} @ {node_info.get('host')}")
            
            if self.on_node_discovered:
                await self._safe_callback(self.on_node_discovered, node_info)
    
    async def discover_nodes(self) -> List[Dict[str, Any]]:
        """发现所有节点"""
        # 发送mDNS查询
        query = {
            "type": "node_query",
            "service": self.SERVICE_TYPE
        }
        
        try:
            self._socket.sendto(
                json.dumps(query).encode(),
                ('<broadcast>', self.port)
            )
        except Exception as e:
            logger.error(f"发送mDNS查询失败: {e}")
        
        # 返回缓存的节点
        with self._lock:
            return list(self.discovered_nodes.values())
    
    async def announce(self, node_info: Dict[str, Any]) -> None:
        """广播节点信息"""
        message = {
            "type": "node_announce",
            "service": self.SERVICE_TYPE,
            "node": node_info
        }
        
        try:
            self._socket.sendto(
                json.dumps(message).encode(),
                ('<broadcast>', self.port)
            )
            logger.debug(f"已广播节点: {node_info.get('name')}")
        except Exception as e:
            logger.error(f"广播节点信息失败: {e}")
    
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


class ManualDiscovery(NodeDiscovery):
    """手动节点发现（通过配置列表）"""
    
    def __init__(self, nodes: Optional[List[Dict[str, Any]]] = None):
        """
        初始化手动发现
        
        Args:
            nodes: 初始节点列表
        """
        self.nodes: Dict[str, Dict[str, Any]] = {}
        
        if nodes:
            for node in nodes:
                self.nodes[node.get("node_id", "")] = node
        
        logger.info(f"手动发现初始化: {len(self.nodes)} 个预配置节点")
    
    async def start(self) -> None:
        """启动（手动发现无需后台服务）"""
        logger.info("手动发现服务已启动")
    
    async def stop(self) -> None:
        """停止"""
        logger.info("手动发现服务已停止")
    
    async def discover_nodes(self) -> List[Dict[str, Any]]:
        """返回预配置的节点"""
        return list(self.nodes.values())
    
    async def announce(self, node_info: Dict[str, Any]) -> None:
        """注册本节点"""
        node_id = node_info.get("node_id")
        if node_id:
            self.nodes[node_id] = node_info
            logger.info(f"节点已注册: {node_info.get('name')}")
    
    def add_node(self, node_info: Dict[str, Any]) -> None:
        """添加节点"""
        node_id = node_info.get("node_id")
        if node_id:
            self.nodes[node_id] = node_info
    
    def remove_node(self, node_id: str) -> bool:
        """移除节点"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False