"""
KCake 集群管理模块

提供节点发现、集群管理、分片路由等功能
"""

__version__ = "0.1.0"
__author__ = "KCake Team"

from .manager import ClusterManager, NodeRole, NodeInfo, ShardInfo
from .discovery import NodeDiscovery, mDNSDiscovery
from .router import ShardRouter

__all__ = [
    "ClusterManager",
    "NodeRole",
    "NodeInfo",
    "ShardInfo",
    "NodeDiscovery",
    "mDNSDiscovery",
    "ShardRouter",
]