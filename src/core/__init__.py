"""
KCake Core - 核心推理引擎

负责任务：
1. 模型加载与分片
2. Token 生成管理
3. KV Cache 管理
4. 与异构调度器协作
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, AsyncIterator, Union
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """支持的设备类型"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal
    VULKAN = "vulkan"
    NPU = "npu"
    TPU = "tpu"
    REMOTE = "remote"  # 分布式节点


@dataclass
class DeviceInfo:
    """设备信息"""
    device_id: str
    device_type: DeviceType
    name: str
    memory_total: int  # bytes
    memory_available: int
    compute_units: int  # GPU cores / CPU threads
    numa_node: Optional[int] = None
    is_edge: bool = False  # 手机等边缘设备


@dataclass
class ModelShard:
    """模型分片信息"""
    shard_id: int
    layer_range: range
    device: DeviceInfo
    size_bytes: int
    loaded: bool = False


@dataclass
class InferenceRequest:
    """推理请求"""
    model: str
    prompt: str
    stream: bool = False
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None


class InferenceEngine:
    """
    核心推理引擎
    
    负责：
    - 模型加载和分片
    - Token 生成
    - KV Cache 管理
    - 与异构调度器协作
    """
    
    def __init__(
        self,
        model_path: str,
        device_map: Optional[Dict[str, Any]] = None,
        max_memory: Optional[Dict[str, int]] = None,
        torch_dtype: str = "float16",
        quantization: Optional[str] = "int4",  # int4, int8, fp8, None
    ):
        self.model_path = model_path
        self.device_map = device_map or "auto"
        self.max_memory = max_memory
        self.torch_dtype = torch_dtype
        self.quantization = quantization
        
        self.model = None
        self.tokenizer = None
        self.device_info: Dict[str, DeviceInfo] = {}
        self.shards: List[ModelShard] = []
        
        logger.info(f"InferenceEngine initialized for model: {model_path}")
    
    async def load_model(self) -> None:
        """加载模型"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # 动态导入以延迟加载
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 确定加载参数
            load_kwargs = {
                "device_map": self.device_map,
                "torch_dtype": getattr(torch, self.torch_dtype),
                "trust_remote_code": True,
            }
            
            if self.max_memory:
                load_kwargs["max_memory"] = self.max_memory
            
            # 量化配置
            if self.quantization:
                from .quantization import get_quantization_config
                load_kwargs["quantization_config"] = get_quantization_config(
                    self.quantization
                )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **load_kwargs
            )
            
            logger.info(f"Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def generate(
        self,
        request: InferenceRequest
    ) -> AsyncIterator[str]:
        """
        生成文本
        
        Args:
            request: 推理请求
            
        Yields:
            生成的 token 字符串
        """
        if not self.model or not self.tokenizer:
            await self.load_model()
        
        # 编码输入
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True
        )
        
        # 确定设备
        device = next(self.model.parameters()).device
        
        # 生成
        with torch.no_grad():
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            generation_kwargs = {
                **inputs,
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "streamer": streamer,
            }
            
            if request.stop:
                # 自定义停止逻辑
                generation_kwargs["stop_strings"] = request.stop
            
            # 多线程生成
            thread = Thread(
                target=self.model.generate,
                kwargs=generation_kwargs
            )
            thread.start()
            
            # 流式输出
            for text in streamer:
                yield text
            
            thread.join()
    
    async def generate_batch(
        self,
        requests: List[InferenceRequest]
    ) -> List[str]:
        """批量生成"""
        results = []
        for req in requests:
            output = ""
            async for token in self.generate(req):
                output += token
            results.append(output)
        return results
    
    def get_device_map(self) -> Dict[str, Any]:
        """获取当前设备映射"""
        return self.device_map
    
    def set_heterogeneous_scheduler(self, scheduler) -> None:
        """设置异构调度器"""
        self.scheduler = scheduler
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.model:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            "model_path": self.model_path,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device_map": self.device_map,
            "dtype": self.torch_dtype,
            "quantization": self.quantization,
        }
    
    async def unload_model(self) -> None:
        """卸载模型释放内存"""
        if self.model:
            del self.model
            self.model = None
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded")
    
    async def health_check(self) -> bool:
        """健康检查"""
        return self.model is not None and self.tokenizer is not None
