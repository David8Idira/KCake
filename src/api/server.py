"""
API服务器模块

基于FastAPI的REST API服务
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============== 请求/响应模型 ==============

class ChatMessage(BaseModel):
    """聊天消息"""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """聊天完成请求"""
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


class CompletionRequest(BaseModel):
    """文本完成请求"""
    model: str
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


class EmbeddingRequest(BaseModel):
    """嵌入请求"""
    model: str = "text-embedding-ada-002"
    input: Union[str, List[str]]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    engine: str


class ErrorResponse(BaseModel):
    """错误响应"""
    error: Dict[str, Any]


# ============== API服务器 ==============

class KcakeAPIServer:
    """KCake API服务器"""
    
    def __init__(
        self,
        inference_engine: Any,
        cluster_manager: Optional[Any] = None,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        """
        初始化API服务器
        
        Args:
            inference_engine: 推理引擎实例
            cluster_manager: 集群管理器（可选）
            host: 监听地址
            port: 监听端口
        """
        self.inference_engine = inference_engine
        self.cluster_manager = cluster_manager
        self.host = host
        self.port = port
        
        # 创建FastAPI应用
        self.app = FastAPI(
            title="KCake API",
            description="异构设备集群化超大规模模型推理引擎API",
            version="0.1.0"
        )
        
        # 配置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 注册路由
        self._register_routes()
        
        # 服务器配置
        self.server_config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        self.server = uvicorn.Server(self.server_config)
        
        logger.info(f"API服务器初始化: {host}:{port}")
    
    def _register_routes(self) -> None:
        """注册API路由"""
        
        # 健康检查
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            engine_info = self.inference_engine.get_model_info()
            return HealthResponse(
                status="healthy",
                version="0.1.0",
                engine="ready" if engine_info.get("is_loaded") else "not_loaded"
            )
        
        # OpenAI兼容API - 聊天完成
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            try:
                # 转换消息为prompt
                prompt = self._messages_to_prompt(request.messages)
                
                # 创建推理请求
                from ..core.inference_engine import InferenceRequest
                inf_request = InferenceRequest(
                    model=request.model,
                    prompt=prompt,
                    stream=request.stream,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    seed=request.seed
                )
                
                # 执行推理
                if request.stream:
                    return StreamingResponse(
                        self._stream_response(inf_request),
                        media_type="application/json"
                    )
                else:
                    response = await self.inference_engine.generate(inf_request)
                    return self._build_chat_completion_response(response, request)
                    
            except Exception as e:
                logger.error(f"聊天完成请求失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # OpenAI兼容API - 文本完成
        @self.app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            try:
                from ..core.inference_engine import InferenceRequest
                inf_request = InferenceRequest(
                    model=request.model,
                    prompt=request.prompt,
                    stream=request.stream,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    seed=request.seed
                )
                
                if request.stream:
                    return StreamingResponse(
                        self._stream_response(inf_request),
                        media_type="application/json"
                    )
                else:
                    response = await self.inference_engine.generate(inf_request)
                    return self._build_completion_response(response, request)
                    
            except Exception as e:
                logger.error(f"文本完成请求失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # OpenAI兼容API - 嵌入
        @self.app.post("/v1/embeddings")
        async def embeddings(request: EmbeddingRequest):
            try:
                # 简化实现
                results = await self._compute_embeddings(request.input)
                return {
                    "object": "list",
                    "data": results,
                    "model": request.model
                }
            except Exception as e:
                logger.error(f"嵌入请求失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # OpenAI兼容API - 模型列表
        @self.app.get("/v1/models")
        async def list_models():
            engine_info = self.inference_engine.get_model_info()
            return {
                "object": "list",
                "data": [
                    {
                        "id": engine_info.get("model_name", "default"),
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "kcake",
                        "permission": [],
                    }
                ]
            }
        
        # Ollama兼容API - 生成
        @self.app.post("/api/generate")
        async def ollama_generate(request: Request):
            try:
                body = await request.json()
                prompt = body.get("prompt", "")
                model = body.get("model", "default")
                stream = body.get("stream", False)
                options = body.get("options", {})
                
                from ..core.inference_engine import InferenceRequest
                inf_request = InferenceRequest(
                    model=model,
                    prompt=prompt,
                    stream=stream,
                    max_tokens=options.get("num_predict", 512),
                    temperature=options.get("temperature", 0.7),
                    top_p=options.get("top_p", 0.9),
                )
                
                if stream:
                    return StreamingResponse(
                        self._ollama_stream_response(inf_request),
                        media_type="application/json"
                    )
                else:
                    response = await self.inference_engine.generate(inf_request)
                    return {
                        "model": model,
                        "response": response.text,
                        "done": True,
                        "context": None,
                        "total_duration": response.tokens_generated * 1000000,  # ns
                        "load_duration": 0,
                        "prompt_eval_count": response.usage.get("prompt_tokens", 0),
                        "prompt_eval_duration": 0,
                        "eval_count": response.tokens_generated,
                        "eval_duration": response.tokens_generated * 1000000,
                    }
                    
            except Exception as e:
                logger.error(f"Ollama生成请求失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Ollama兼容API - 聊天
        @self.app.post("/api/chat")
        async def ollama_chat(request: Request):
            try:
                body = await request.json()
                messages = body.get("messages", [])
                model = body.get("model", "default")
                stream = body.get("stream", False)
                
                # 转换消息为prompt
                prompt = self._ollama_messages_to_prompt(messages)
                
                from ..core.inference_engine import InferenceRequest
                inf_request = InferenceRequest(
                    model=model,
                    prompt=prompt,
                    stream=stream,
                    max_tokens=512,
                )
                
                if stream:
                    return StreamingResponse(
                        self._ollama_stream_response(inf_request),
                        media_type="application/json"
                    )
                else:
                    response = await self.inference_engine.generate(inf_request)
                    return {
                        "model": model,
                        "message": {"role": "assistant", "content": response.text},
                        "done": True,
                        "total_duration": response.tokens_generated * 1000000,
                        "load_duration": 0,
                        "prompt_eval_count": response.usage.get("prompt_tokens", 0),
                        "eval_count": response.tokens_generated,
                        "eval_duration": response.tokens_generated * 1000000,
                    }
                    
            except Exception as e:
                logger.error(f"Ollama聊天请求失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Ollama兼容API - 模型列表
        @self.app.get("/api/tags")
        async def ollama_list_models():
            engine_info = self.inference_engine.get_model_info()
            return {
                "models": [
                    {
                        "name": engine_info.get("model_name", "default"),
                        "modified_at": time.time(),
                        "size": 0,
                        "digest": "sha256:0000000000",
                    }
                ]
            }
        
        # 集群状态
        @self.app.get("/cluster/status")
        async def cluster_status():
            if self.cluster_manager is None:
                return {"error": "集群管理器未初始化"}
            return self.cluster_manager.get_cluster_status()
        
        logger.info("API路由注册完成")
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """将聊天消息转换为prompt"""
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
        prompt += "Assistant: "
        return prompt
    
    def _ollama_messages_to_prompt(self, messages: List[Dict]) -> str:
        """将Ollama格式消息转换为prompt"""
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
    
    async def _stream_response(self, request: Any):
        """流式响应生成器"""
        try:
            full_text = ""
            async for text in self.inference_engine.generate_stream(request):
                full_text += text
                
                # 构建SSE格式
                chunk = {
                    "id": f"chatcmpl-{int(time.time() * 1000)}",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "delta": {"content": text},
                        "finish_reason": None
                    }]
                }
                
                yield f"data: {self._json_dumps(chunk)}\n\n"
            
            # 发送完成信号
            final_chunk = {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {self._json_dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"流式响应生成失败: {e}")
            yield f"data: {self._json_dumps({'error': str(e)})}\n\n"
    
    async def def _ollama_stream_response(self, request: Any):
        """Ollama流式响应生成器"""
        try:
            async for text in self.inference_engine.generate_stream(request):
                chunk = {
                    "model": request.model,
                    "response": text,
                    "done": False
                }
                yield f"data: {self._json_dumps(chunk)}\n\n"
            
            # 发送完成信号
            final_chunk = {
                "model": request.model,
                "done": True,
                "response": ""
            }
            yield f"data: {self._json_dumps(final_chunk)}\n\n"
            
        except Exception as e:
            logger.error(f"Ollama流式响应失败: {e}")
            yield f"data: {self._json_dumps({'error': str(e)})}\n\n"
    
    def _build_chat_completion_response(self, response: Any, request: Any) -> Dict:
        """构建聊天完成响应"""
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
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
    
    def _build_completion_response(self, response: Any, request: Any) -> Dict:
        """构建文本完成响应"""
        return {
            "id": f"cmpl-{int(time.time() * 1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
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
    
    async def _compute_embeddings(self, texts: Union[str, List[str]]) -> List[Dict]:
        """计算嵌入向量（简化实现）"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        for i, text in enumerate(texts):
            # 简化实现：返回随机向量
            import random
            embedding = [random.random() for _ in range(1536)]
            
            results.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i
            })
        
        return results
    
    def _json_dumps(self, obj: Any) -> str:
        """安全JSON序列化"""
        import json
        return json.dumps(obj, ensure_ascii=False)
    
    async def start(self) -> None:
        """启动服务器"""
        logger.info(f"启动API服务器: {self.host}:{self.port}")
        await self.server.serve()
    
    async def stop(self) -> None:
        """停止服务器"""
        logger.info("停止API服务器")
        self.server.should_exit = True