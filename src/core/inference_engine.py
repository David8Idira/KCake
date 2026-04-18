"""
推理引擎核心模块

提供模型加载、推理生成、KV缓存管理等核心功能
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, AsyncGenerator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """推理请求数据结构"""
    model: str
    prompt: str
    stream: bool = False
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


@dataclass
class InferenceResponse:
    """推理响应数据结构"""
    text: str
    tokens_generated: int
    finish_reason: str
    model: str
    usage: Dict[str, int]


class InferenceEngine:
    """核心推理引擎"""
    
    def __init__(self, device: str = "cpu"):
        """
        初始化推理引擎
        
        Args:
            device: 设备类型 (cpu, cuda, metal, vulkan)
        """
        self.device = device
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model_name: Optional[str] = None
        self.is_loaded = False
        
        # KV缓存管理
        self.kv_cache = {}
        self.max_cache_size = 100  # 最大缓存条目数
        
        logger.info(f"初始化推理引擎，设备: {device}")
    
    async def load_model(self, model_name: str, **kwargs) -> bool:
        """
        加载模型
        
        Args:
            model_name: 模型名称或路径
            **kwargs: 额外参数 (dtype, quantization, device_map等)
            
        Returns:
            bool: 是否加载成功
        """
        try:
            logger.info(f"开始加载模型: {model_name}")
            
            # 设置加载参数
            load_kwargs = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "device_map": "auto" if self.device != "cpu" else None,
                "trust_remote_code": True,
            }
            
            # 合并用户参数
            load_kwargs.update(kwargs)
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            # 如果指定了设备，移动模型到目标设备
            if self.device != "auto" and self.device != "cpu":
                try:
                    self.model = self.model.to(self.device)
                except Exception as e:
                    logger.warning(f"无法移动模型到设备 {self.device}: {e}")
            
            self.model_name = model_name
            self.is_loaded = True
            
            logger.info(f"模型加载成功: {model_name}")
            logger.info(f"模型参数量: {self.model.num_parameters():,}")
            
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
            return False
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        生成文本
        
        Args:
            request: 推理请求
            
        Returns:
            InferenceResponse: 推理响应
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        try:
            logger.info(f"开始生成文本，模型: {self.model_name}, 最大token数: {request.max_tokens}")
            
            # 编码输入
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # 移动输入到设备
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成配置
            generate_kwargs = {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "do_sample": request.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # 设置随机种子
            if request.seed is not None:
                torch.manual_seed(request.seed)
            
            # 执行生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generate_kwargs
                )
            
            # 解码输出
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            # 检查停止符
            finish_reason = "length"
            if request.stop:
                for stop_seq in request.stop:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        finish_reason = "stop"
                        break
            
            # 构建响应
            response = InferenceResponse(
                text=generated_text,
                tokens_generated=len(generated_tokens),
                finish_reason=finish_reason,
                model=self.model_name or "unknown",
                usage={
                    "prompt_tokens": inputs["input_ids"].shape[1],
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": inputs["input_ids"].shape[1] + len(generated_tokens)
                }
            )
            
            logger.info(f"生成完成，生成token数: {response.tokens_generated}, 完成原因: {response.finish_reason}")
            
            return response
            
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            raise
    
    async def generate_stream(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """
        流式生成文本
        
        Args:
            request: 推理请求
            
        Yields:
            str: 生成的文本片段
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        try:
            logger.info(f"开始流式生成，模型: {self.model_name}")
            
            # 编码输入
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # 移动输入到设备
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成配置
            generate_kwargs = {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "do_sample": request.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "streamer": None,  # 这里可以集成自定义streamer
            }
            
            # 设置随机种子
            if request.seed is not None:
                torch.manual_seed(request.seed)
            
            # 执行流式生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generate_kwargs
                )
            
            # 流式解码输出
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            
            # 逐步解码并yield
            decoded_text = ""
            for i in range(len(generated_tokens)):
                token = generated_tokens[i:i+1]
                new_text = self.tokenizer.decode(token, skip_special_tokens=True)
                
                # 检查是否遇到停止符
                if request.stop:
                    for stop_seq in request.stop:
                        if stop_seq in decoded_text + new_text:
                            # 遇到停止符，停止生成
                            yield decoded_text
                            return
                
                decoded_text += new_text
                yield new_text
                
                # 模拟流式延迟
                await asyncio.sleep(0.01)
            
            logger.info(f"流式生成完成，总token数: {len(generated_tokens)}")
            
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            raise
    
    async def unload_model(self) -> None:
        """卸载模型"""
        if self.model:
            # 清理模型
            self.model = None
            self.tokenizer = None
            self.model_name = None
            self.is_loaded = False
            
            # 清理缓存
            self.kv_cache.clear()
            
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("模型已卸载")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "model_type": self.model.__class__.__name__ if self.model else None,
            "num_parameters": self.model.num_parameters() if self.model else 0,
            "kv_cache_size": len(self.kv_cache),
        }
        
        return info
    
    def clear_cache(self) -> None:
        """清理KV缓存"""
        self.kv_cache.clear()
        logger.info("KV缓存已清理")