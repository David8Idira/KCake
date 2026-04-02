"""
KCake API Server - API 服务层

提供：
1. OpenAI 兼容 API
2. Ollama 兼容 API
3. RESTful 接口
4. WebSocket 流式响应
"""

import asyncio
import logging
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, AsyncIterator, Union
from enum import Enum
import uuid
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)


# ============== 请求/响应模型 ==============

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None


@dataclass
class ChatMessage:
    role: str
    content: str


# ============== API Server ==============

class APIServer:
    """
    API 服务器
    
    提供 OpenAI 和 Ollama 兼容的 API 接口
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        engine = None,  # InferenceEngine
        cluster_manager = None,  # ClusterManager
    ):
        self.host = host
        self.port = port
        self.engine = engine
        self.cluster_manager = cluster_manager
        
        self.app = FastAPI(
            title="KCake API",
            description="异构设备集群化超大规模模型推理引擎 API",
            version="0.1.0",
        )
        
        self._setup_middleware()
        self._setup_routes()
        
        logger.info(f"APIServer initialized on {host}:{port}")
    
    def _setup_middleware(self) -> None:
        """设置中间件"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self) -> None:
        """设置路由"""
        # 健康检查
        self.app.get("/health")(self.health_check)
        self.app.get("/v1/models")(self.list_models)
        
        # OpenAI 兼容 API
        self.app.post("/v1/chat/completions")(self.chat_completions)
        self.app.post("/v1/completions")(self.completions)
        self.app.post("/v1/embeddings")(self.embeddings)
        
        # Ollama 兼容 API
        self.app.post("/api/generate")(self.ollama_generate)
        self.app.post("/api/chat")(self.ollama_chat)
        self.app.get("/api/tags")(self.ollama_tags)
        self.app.post("/api/pull")(self.ollama_pull)
        
        # 集群管理 API
        self.app.get("/cluster/status")(self.cluster_status)
        self.app.get("/cluster/nodes")(self.cluster_nodes)
        
        # Web UI
        self.app.get("/")(self.web_ui)
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        engine_healthy = await self.engine.health_check() if self.engine else True
        
        return {
            "status": "healthy" if engine_healthy else "degraded",
            "version": "0.1.0",
            "engine": "ready" if engine_healthy else "not_ready",
        }
    
    async def list_models(self) -> Dict[str, Any]:
        """列出可用模型"""
        models = []
        
        if self.engine:
            info = self.engine.get_model_info()
            models.append({
                "id": info.get("model_path", "unknown"),
                "object": "model",
                "created": 1700000000,
                "owned_by": "system",
                "permission": [],
            })
        
        return {
            "object": "list",
            "data": models,
        }
    
    async def chat_completions(
        self,
        request: ChatCompletionRequest,
    ) -> Union[JSONResponse, StreamingResponse]:
        """
        OpenAI 兼容的聊天完成接口
        """
        if not self.engine:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        
        # 构建提示词
        prompt = self._build_prompt_from_messages(request.messages)
        
        # 创建推理请求
        from ..core import InferenceRequest
        inf_request = InferenceRequest(
            model=request.model,
            prompt=prompt,
            stream=request.stream,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=[request.stop] if isinstance(request.stop, str) else request.stop,
        )
        
        if request.stream:
            return StreamingResponse(
                self._stream_chat_response(inf_request, request.model),
                media_type="text/event-stream",
            )
        else:
            # 非流式响应
            full_response = ""
            async for token in self.engine.generate(inf_request):
                full_response += token
            
            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": 1700000000,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_response,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": len(full_response),
                    "total_tokens": len(prompt) + len(full_response),
                },
            })
    
    async def _stream_chat_response(
        self,
        request,
        model: str,
    ) -> AsyncIterator[str]:
        """流式聊天响应"""
        choice_data = {
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }
        
        async for token in self.engine.generate(request):
            choice_data["message"]["content"] += token
            
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model,
                "choices": [choice_data],
            }
            
            yield f"data: {json.dumps(chunk)}\n\n"
        
        choice_data["finish_reason"] = "stop"
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    def _build_prompt_from_messages(self, messages: List[Message]) -> str:
        """从消息列表构建提示词"""
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    async def completions(self, request: Request) -> JSONResponse:
        """OpenAI 兼容的完成接口"""
        # 类似 chat_completions 但处理纯文本输入
        body = await request.json()
        
        prompt = body.get("prompt", "")
        model = body.get("model", "default")
        max_tokens = body.get("max_tokens", 2048)
        stream = body.get("stream", False)
        
        # 简化实现
        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": 1700000000,
            "model": model,
            "choices": [{
                "text": "Completion response",
                "index": 0,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": 10,
                "total_tokens": len(prompt) + 10,
            },
        })
    
    async def embeddings(self, request: Request) -> JSONResponse:
        """OpenAI 兼容的嵌入接口"""
        body = await request.json()
        
        return JSONResponse({
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": [0.0] * 768,  # 占位符
                "index": 0,
            }],
            "model": body.get("model", "text-embedding-3-small"),
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10,
            },
        })
    
    async def ollama_generate(self, request: OllamaGenerateRequest) -> Union[JSONResponse, StreamingResponse]:
        """Ollama 兼容的生成接口"""
        if not self.engine:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        
        prompt = request.prompt
        if request.system:
            prompt = f"System: {request.system}\n\n{prompt}"
        
        from ..core import InferenceRequest
        inf_request = InferenceRequest(
            model=request.model,
            prompt=prompt,
            stream=request.stream,
            max_tokens=request.options.get("num_predict", 2048) if request.options else 2048,
            temperature=request.options.get("temperature", 0.7) if request.options else 0.7,
        )
        
        if request.stream:
            return StreamingResponse(
                self._stream_ollama_response(inf_request, request.model),
                media_type="text/event-stream",
            )
        else:
            full_response = ""
            async for token in self.engine.generate(inf_request):
                full_response += token
            
            return JSONResponse({
                "model": request.model,
                "response": full_response,
                "done": True,
            })
    
    async def _stream_ollama_response(
        self,
        request,
        model: str,
    ) -> AsyncIterator[str]:
        """流式 Ollama 响应"""
        async for token in self.engine.generate(request):
            response = {
                "model": model,
                "response": token,
                "done": False,
            }
            yield f"data: {json.dumps(response)}\n\n"
        
        yield f"data: {json.dumps({'model': model, 'done': True})}\n\n"
    
    async def ollama_chat(self, request: Request) -> JSONResponse:
        """Ollama 兼容的聊天接口"""
        body = await request.json()
        messages = body.get("messages", [])
        
        prompt = self._build_prompt_from_messages([
            Message(role=m["role"], content=m["content"]) for m in messages
        ])
        
        return JSONResponse({
            "model": body.get("model", "llama3.1"),
            "message": {
                "role": "assistant",
                "content": "Chat response",
            },
            "done": True,
        })
    
    async def ollama_tags(self) -> JSONResponse:
        """Ollama 列出可用模型"""
        models = await self.list_models()
        
        return JSONResponse({
            "models": [
                {
                    "name": m["id"],
                    "model": m["id"],
                    "size": 0,
                    "modified_at": "2024-01-01T00:00:00Z",
                }
                for m in models.get("data", [])
            ]
        })
    
    async def ollama_pull(self, request: Request) -> JSONResponse:
        """Ollama 拉取模型 (占位)"""
        body = await request.json()
        name = body.get("name", "")
        
        return JSONResponse({
            "status": "success",
            "message": f"Model {name} pulled successfully",
        })
    
    async def cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        if not self.cluster_manager:
            return {"error": "Cluster manager not initialized"}
        
        return self.cluster_manager.get_cluster_status()
    
    async def cluster_nodes(self) -> Dict[str, Any]:
        """获取集群节点列表"""
        if not self.cluster_manager:
            return {"error": "Cluster manager not initialized"}
        
        status = self.cluster_manager.get_cluster_status()
        return status.get("nodes", {})
    
    async def web_ui(self) -> Dict[str, str]:
        """Web UI 入口"""
        return {"message": "KCake Web UI - see /docs for API documentation"}
    
    async def start(self) -> None:
        """启动 API 服务器"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        
        logger.info(f"Starting API server on {self.host}:{self.port}")
        await server.serve()
    
    def run(self) -> None:
        """同步启动"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
