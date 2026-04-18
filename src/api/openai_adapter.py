"""
OpenAI API适配器

提供OpenAI API兼容接口
"""

import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
import time

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    """OpenAI API适配器"""
    
    def __init__(self, inference_engine: Any):
        """
        初始化OpenAI适配器
        
        Args:
            inference_engine: 推理引擎实例
        """
        self.engine = inference_engine
        
        # API版本
        self.api_version = "v1"
        
        logger.info("OpenAI适配器初始化完成")
    
    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        聊天完成接口
        
        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            top_p: top-p采样
            max_tokens: 最大token数
            stream: 是否流式
            stop: 停止符列表
            
        Returns:
            Dict: 响应数据
        """
        # 转换消息为prompt
        prompt = self._format_messages(messages)
        
        # 构建请求
        from ..core.inference_engine import InferenceRequest
        request = InferenceRequest(
            model=model,
            prompt=prompt,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )
        
        # 执行推理
        if stream:
            return await self._stream_chat_completions(request, model, messages)
        else:
            response = await self.engine.generate(request)
            return self._format_chat_response(response, model)
    
    async def completions(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        文本完成接口
        
        Args:
            model: 模型名称
            prompt: 提示文本
            temperature: 温度参数
            top_p: top-p采样
            max_tokens: 最大token数
            stream: 是否流式
            stop: 停止符列表
            
        Returns:
            Dict: 响应数据
        """
        from ..core.inference_engine import InferenceRequest
        request = InferenceRequest(
            model=model,
            prompt=prompt,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )
        
        if stream:
            return await self._stream_completions(request, model, prompt)
        else:
            response = await self.engine.generate(request)
            return self._format_completion_response(response, model, prompt)
    
    async def embeddings(
        self,
        model: str,
        input: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        嵌入接口
        
        Args:
            model: 模型名称
            input: 输入文本或文本列表
            
        Returns:
            Dict: 响应数据
        """
        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input
        
        # 简化实现
        import random
        data = []
        
        for i, text in enumerate(inputs):
            embedding = [random.random() for _ in range(1536)]
            data.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i
            })
        
        return {
            "object": "list",
            "data": data,
            "model": model
        }
    
    def list_models(self) -> Dict[str, Any]:
        """获取模型列表"""
        info = self.engine.get_model_info()
        
        return {
            "object": "list",
            "data": [
                {
                    "id": info.get("model_name", "default"),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "kcake",
                    "permission": [],
                }
            ]
        }
    
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
    
    def _format_chat_response(
        self,
        response: Any,
        model: str
    ) -> Dict[str, Any]:
        """格式化聊天响应"""
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.text
                },
                "finish_reason": response.finish_reason
            }],
            "usage": {
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
                "completion_tokens": response.tokens_generated,
                "total_tokens": response.usage.get("total_tokens", 0)
            }
        }
    
    def _format_completion_response(
        self,
        response: Any,
        model: str,
        prompt: str
    ) -> Dict[str, Any]:
        """格式化文本完成响应"""
        return {
            "id": f"cmpl-{int(time.time() * 1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "text": response.text,
                "index": 0,
                "finish_reason": response.finish_reason
            }],
            "usage": {
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
                "completion_tokens": response.tokens_generated,
                "total_tokens": response.usage.get("total_tokens", 0)
            }
        }
    
    async def _stream_chat_completions(
        self,
        request: Any,
        model: str,
        messages: List[Dict]
    ) -> AsyncGenerator[str, None]:
        """流式聊天完成"""
        chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
        
        async for text in self.engine.generate_stream(request):
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {"content": text},
                    "finish_reason": None
                }]
            }
            yield f"data: {self._json_encode(chunk)}\n\n"
        
        # 发送最终块
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {self._json_encode(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    async def _stream_completions(
        self,
        request: Any,
        model: str,
        prompt: str
    ) -> AsyncGenerator[str, None]:
        """流式文本完成"""
        chunk_id = f"cmpl-{int(time.time() * 1000)}"
        
        async for text in self.engine.generate_stream(request):
            chunk = {
                "id": chunk_id,
                "object": "text_completion",
                "choices": [{
                    "text": text,
                    "finish_reason": None
                }]
            }
            yield f"data: {self._json_encode(chunk)}\n\n"
        
        # 发送最终块
        final_chunk = {
            "id": chunk_id,
            "object": "text_completion",
            "choices": [{
                "text": "",
                "finish_reason": "stop"
            }]
        }
        yield f"data: {self._json_encode(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    def _json_encode(self, obj: Any) -> str:
        """JSON编码"""
        import json
        return json.dumps(obj, ensure_ascii=False)


# 类型注解
from typing import Union
