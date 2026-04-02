"""
KCake - 异构设备集群化超大规模模型推理引擎

融合 ktransformers (高性能CPU-GPU异构计算) 与 cake (分布式设备集群推理) 的优势
"""

__version__ = "0.1.0"
__author__ = "KCake Team"

from .core import InferenceEngine
from .heterogeneous import HeteroScheduler
from .cluster import ClusterManager
from .api import APIServer

__all__ = [
    "InferenceEngine",
    "HeteroScheduler", 
    "ClusterManager",
    "APIServer",
]
