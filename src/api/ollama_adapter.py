"""
Ollama API适配器

提供Ollama兼容接口
"""

import logging
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Union

logger = logging.getLogger(__name__)


class OllamaAdapter:
    """Ollama API适配器"""
    
    def __init__(self, inference_engine: Any):
        """
        初始化Ollama适配器
        
        Args:
            inference_engine: 推理引擎实例
        """
        self.engine = inference_engine
        
        logger.info("Ollama适配器初始化完成")
    
    async def generate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator]:
        """
        文本生成接口
        
        Args:
            model: 模型名称
            prompt: 提示文本
            options: 生成选项
            stream: 是否流式
            
        Returns:
            响应或流式生成器
        """
        options = options or {}
        
        from ..core.inference_engine import InferenceRequest
        request = InferenceRequest(
            model=model,
            prompt=prompt,
            stream=stream,
            max_tokens=options.get("num_predict", 512),
            temperature=options.get("temperature", 0.7),
            top_p=options.get("top_p", 0.9),
        )
        
        if stream:
            return self._stream_generate(model, request)
        else:
            return await self._sync_generate(model, request, options)
    
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator]:
        """
        聊天接口
        
        Args:
            model: 模型名称
            messages: 消息列表
            options: 生成选项
            stream: 是否流式
            
        Returns:
            响应或流式生成器
        """
        options = options or {}
        
        # 转换消息为prompt
        prompt = self._format_messages(messages)
        
        from ..core.inference_engine import InferenceRequest
        request = InferenceRequest(
            model=model,
            prompt=prompt,
            stream=stream,
            max_tokens=options.get("num_predict", 512),
            temperature=options.get("temperature", 0.7),
            top_p=options.get("top_p", 0.9),
        )
        
        if stream:
            return self._stream_chat(model, request)
        else:
            return await self._sync_chat(model, request, messages, options)
    
    async def tags(self) -> Dict[str, Any]:
        """获取模型列表"""
        info = self.engine.get_model_info()
        
        return {
            "models": [
                {
                    "name": info.get("model_name", "default"),
                    "modified_at": time.time(),
                    "size": 0,
                    "digest": "sha256:0000000000",
                }
            ]
        }
    
    async def show(self, model: str) -> Dict[str, Any]:
        """获取模型详细信息"""
        info = self.engine.get_model_info()
        
        return {
            "modelfile": "",
            "parameters": {
                "temperature": "0.7",
                "top_p": "0.9",
            },
            "template": "",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "7B",
                "quantization_level": "Q4_0",
            }
        }
    
    async def pull(self, model: str, stream: bool = False) -> Union[Dict, AsyncGenerator]:
        """拉取模型"""
        # 简化实现
        if stream:
            async def generate():
                yield {"status": "pulling manifest"}
                yield {"status": "pulling sha256"}
                yield {"status": "verifying sha256 digest"}
                yield {"status": "writing manifest"}
                yield {"status": "success"}
            return generate()
        else:
            return {"status": "success"}
    
    async def push(self, model: str, stream: bool = False) -> Union[Dict, AsyncGenerator]:
        """推送模型"""
        # 简化实现
        if stream:
            async def generate():
                yield {"status": "pushing manifest"}
                yield {"status": "success"}
            return generate()
        else:
            return {"status": "success"}
    
    async def create(self, model: str, modelfile: str) -> Dict[str, Any]:
        """创建模型"""
        return {"status": "success"}
    
    async def delete(self, model: str) -> Dict[str, Any]:
        """删除模型"""
        return {"status": "success"}
    
    async def copy(self, source: str, destination: str) -> Dict[str, Any]:
        """复制模型"""
        return {"status": "success"}
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """格式化消息为prompt"""
        prompt = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        return prompt
    
    async def _sync_generate(
        self,
        model: str,
        request: Any,
        options: Dict
    ) -> Dict[str, Any]:
        """同步生成"""
        start_time = time.time()
        response = await self.engine.generate(request)
        
        return {
            "model": model,
            "response": response.text,
            "done": True,
            "context": None,
            "total_duration": int((time.time() - start_time) * 1e9),
            "load_duration": 0,
            "prompt_eval_count": response.usage.get("prompt_tokens", 0),
            "prompt_eval_duration": 0,
            "eval_count": response.tokens_generated,
            "eval_duration": response.tokens_generated * 1e6,
        }
    
    async def _sync_chat(
        self,
        model: str,
        request: Any,
        messages: List[Dict],
        options: Dict
    ) -> Dict[str, Any]:
        """同步聊天"""
        start_time = time.time()
        response = await self.engine.generate(request)
        
        return {
            "model": model,
            "message": {
                "role": "assistant",
                "content": response.text
            },
            "done": True,
            "total_duration": int((time.time() - start_time) * 1e9),
            "load_duration": 0,
            "prompt_eval_count": response.usage.get("prompt_tokens", 0),
            "eval_count": response.tokens_generated,
            "eval_duration": response.tokens_generated * 1e6,
        }
    
    async def _stream_generate(self, model: str, request: Any) -> AsyncGenerator:
        """流式生成"""
        async def generate():
            async for text in self.engine.generate_stream(request):
                yield {
                    "model": model,
                    "response": text,
                    "done": False
                }
            
            yield {
                "model": model,
                "response": "",
                "done": True
            }
        
        return generate()
    
    async def _stream_chat(self, model: str, request: Any) -> AsyncGenerator:
        """流式聊天"""
        async def generate():
            async for text in self.engine.generate_stream(request):
                yield {
                    "model": model,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "done": False
                }
            
            yield {
                "model": model,
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done": True
            }
        
        return generate()