"""
模型加载器模块

负责模型的下载、加载、卸载、缓存管理
"""

import asyncio
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    path: str
    size_bytes: int
    num_parameters: int
    dtype: str
    quantization: Optional[str] = None
    is_cached: bool = False


class ModelLoader:
    """模型加载器"""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        """
        初始化模型加载器
        
        Args:
            cache_dir: 模型缓存目录
            hf_token: HuggingFace访问令牌
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        self.hf_token = hf_token
        self.loaded_models: Dict[str, ModelInfo] = {}
        
        logger.info(f"模型加载器初始化，缓存目录: {self.cache_dir}")
    
    async def download_model(self, model_name: str, **kwargs) -> str:
        """
        下载模型
        
        Args:
            model_name: 模型名称
            **kwargs: 额外参数
            
        Returns:
            str: 模型本地路径
        """
        try:
            logger.info(f"开始下载模型: {model_name}")
            
            # 这里应该调用 huggingface_hub 进行下载
            # 由于是模拟实现，返回路径
            model_path = os.path.join(self.cache_dir, "models", model_name)
            
            # 检查是否已缓存
            if os.path.exists(model_path):
                logger.info(f"模型已缓存: {model_path}")
                return model_path
            
            # 模拟下载过程
            logger.info(f"模型下载中 (模拟): {model_name}")
            await asyncio.sleep(0.1)  # 模拟异步操作
            
            logger.info(f"模型下载完成: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            raise
    
    async def load_model_to_engine(self, engine, model_name: str, **kwargs) -> bool:
        """
        加载模型到推理引擎
        
        Args:
            engine: 推理引擎实例
            model_name: 模型名称
            **kwargs: 额外参数
            
        Returns:
            bool: 是否加载成功
        """
        try:
            logger.info(f"开始加载模型到引擎: {model_name}")
            
            # 加载模型
            success = await engine.load_model(model_name, **kwargs)
            
            if success:
                # 记录已加载模型
                self.loaded_models[model_name] = ModelInfo(
                    name=model_name,
                    path=os.path.join(self.cache_dir, "models", model_name),
                    size_bytes=0,  # 模拟
                    num_parameters=0,  # 模拟
                    dtype=kwargs.get("dtype", "float16"),
                    quantization=kwargs.get("quantization"),
                    is_cached=True
                )
                logger.info(f"模型加载到引擎成功: {model_name}")
            else:
                logger.error(f"模型加载到引擎失败: {model_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"加载模型到引擎异常: {e}")
            return False
    
    async def unload_model_from_engine(self, engine, model_name: str) -> bool:
        """
        从推理引擎卸载模型
        
        Args:
            engine: 推理引擎实例
            model_name: 模型名称
            
        Returns:
            bool: 是否卸载成功
        """
        try:
            logger.info(f"开始卸载模型: {model_name}")
            
            await engine.unload_model()
            
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
            
            logger.info(f"模型卸载成功: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"卸载模型异常: {e}")
            return False
    
    def get_cached_models(self) -> Dict[str, ModelInfo]:
        """获取所有已缓存的模型"""
        return self.loaded_models.copy()
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """获取模型路径"""
        model_path = os.path.join(self.cache_dir, "models", model_name)
        if os.path.exists(model_path):
            return model_path
        return None
    
    async def clear_cache(self, model_name: Optional[str] = None) -> bool:
        """
        清理模型缓存
        
        Args:
            model_name: 模型名称，None表示清理所有
            
        Returns:
            bool: 是否清理成功
        """
        try:
            if model_name:
                # 清理指定模型
                model_path = os.path.join(self.cache_dir, "models", model_name)
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                    logger.info(f"已清理模型缓存: {model_name}")
            else:
                # 清理所有缓存
                cache_path = os.path.join(self.cache_dir, "models")
                if os.path.exists(cache_path):
                    shutil.rmtree(cache_path)
                    logger.info("已清理所有模型缓存")
            
            return True
            
        except Exception as e:
            logger.error(f"清理缓存异常: {e}")
            return False
    
    def get_cache_size(self) -> int:
        """获取缓存大小（字节）"""
        total_size = 0
        cache_path = os.path.join(self.cache_dir, "models")
        
        if os.path.exists(cache_path):
            for dirpath, dirnames, filenames in os.walk(cache_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        
        return total_size