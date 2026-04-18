"""
REST API路由

自定义REST接口
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter

logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter()


def setup_routes(app: Any, engine: Any, cluster_manager: Any = None) -> None:
    """
    设置自定义REST路由
    
    Args:
        app: FastAPI应用实例
        engine: 推理引擎实例
        cluster_manager: 集群管理器实例
    """
    
    @router.get("/v1/info")
    async def get_info():
        """获取系统信息"""
        return {
            "name": "KCake",
            "version": "0.1.0",
            "description": "异构设备集群化超大规模模型推理引擎",
            "features": [
                "CPU-GPU混合推理",
                "MoE专家调度",
                "分布式集群管理",
                "OpenAI兼容API",
                "Ollama兼容API"
            ]
        }
    
    @router.get("/v1/stats")
    async def get_stats():
        """获取运行统计"""
        return {
            "uptime": 0,  # TODO: 实现
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
        }
    
    @router.post("/v1/cache/clear")
    async def clear_cache():
        """清理缓存"""
        engine.clear_cache()
        return {"status": "success", "message": "缓存已清理"}
    
    @router.get("/v1/health/detailed")
    async def detailed_health():
        """详细健康检查"""
        model_info = engine.get_model_info()
        
        return {
            "status": "healthy",
            "components": {
                "engine": {
                    "loaded": model_info.get("is_loaded", False),
                    "model": model_info.get("model_name", "none")
                },
                "cluster": {
                    "enabled": cluster_manager is not None,
                    "nodes": cluster_manager.get_cluster_status() if cluster_manager else {}
                }
            }
        }
    
    logger.info("REST路由设置完成")
