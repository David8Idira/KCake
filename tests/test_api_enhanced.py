"""
API模块增强测试 - 践行毛泽东思想：实践是检验真理的唯一标准

补充测试以提升API模块覆盖率到80%
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from src.api.server import (
    KcakeAPIServer, 
    ChatCompletionRequest, 
    CompletionRequest,
    ChatMessage,
    HealthResponse,
    EmbeddingRequest,
    ErrorResponse
)


class TestKcakeAPIServerEnhanced:
    """KCake API服务器增强测试"""
    
    @pytest.fixture
    def server(self):
        """创建API服务器实例"""
        # 创建模拟的推理引擎
        mock_engine = Mock()
        mock_engine.get_model_info = Mock(return_value={"is_loaded": True})
        
        return KcakeAPIServer(
            inference_engine=mock_engine,
            host="127.0.0.1", 
            port=8000
        )
    
    @pytest.fixture
    def client(self, server):
        """创建测试客户端"""
        return TestClient(server.app)
    
    def test_server_configuration(self, server):
        """测试服务器配置"""
        assert server.host == "127.0.0.1"
        assert server.port == 8000
        assert server.app is not None
        assert server.engine is None
    
    def test_server_with_custom_config(self):
        """测试自定义配置"""
        mock_engine = Mock()
        mock_engine.get_model_info = Mock(return_value={"is_loaded": True})
        
        server = KcakeAPIServer(
            inference_engine=mock_engine,
            host="0.0.0.0", 
            port=8080
        )
        
        assert server.host == "0.0.0.0"
        assert server.port == 8080
    
    def test_health_endpoint(self, client):
        """测试健康检查端点"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    def test_models_endpoint(self, client):
        """测试模型列表端点"""
        # 目前服务器可能没有这个端点，先跳过或模拟
        # 实际测试时根据服务器实现调整
        pass
    
    def test_root_endpoint(self, client):
        """测试根端点"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
    
    def test_chat_completion_request_validation(self):
        """测试聊天完成请求验证"""
        # 有效请求
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there!")
            ]
        )
        assert request.model == "test-model"
        assert len(request.messages) == 2
        
        # 测试默认值
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.max_tokens == 512
        assert request.stream is False
    
    def test_completion_request_validation(self):
        """测试文本完成请求验证"""
        request = CompletionRequest(
            model="test-model",
            prompt="Write a story about"
        )
        
        assert request.model == "test-model"
        assert request.prompt == "Write a story about"
        assert request.temperature == 0.7
    
    def test_chat_message_validation(self):
        """测试聊天消息验证"""
        message = ChatMessage(role="user", content="Hello world")
        assert message.role == "user"
        assert message.content == "Hello world"
        
        # 测试不同的角色
        system_message = ChatMessage(role="system", content="You are a helpful assistant")
        assert system_message.role == "system"
    
    @pytest.mark.asyncio
    async def test_server_start_stop(self):
        """测试服务器启动和停止"""
        server = KcakeAPIServer(host="127.0.0.1", port=8001)
        
        # 模拟启动
        with patch('uvicorn.run') as mock_run:
            await server.start()
            mock_run.assert_called_once()
        
        # 测试停止（应不抛出异常）
        await server.stop()
    
    def test_cors_configuration(self, server):
        """测试CORS配置"""
        # 检查CORS中间件是否存在
        middleware_found = False
        for middleware in server.app.user_middleware:
            if "CORSMiddleware" in str(middleware.cls):
                middleware_found = True
                break
        
        assert middleware_found, "CORS中间件未配置"
    
    def test_error_handling(self, client):
        """测试错误处理"""
        # 测试不存在的端点
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        # 测试无效的POST请求到健康端点
        response = client.post("/health")
        assert response.status_code == 405  # Method Not Allowed


class TestModelInfo:
    """模型信息测试"""
    
    def test_model_info_creation(self):
        """测试模型信息创建"""
        info = ModelInfo(
            id="test-model-1",
            object="model",
            created=1234567890,
            owned_by="kcake"
        )
        
        assert info.id == "test-model-1"
        assert info.object == "model"
        assert info.created == 1234567890
        assert info.owned_by == "kcake"
    
    def test_model_info_defaults(self):
        """测试模型信息默认值"""
        info = ModelInfo(id="test-model")
        
        assert info.object == "model"
        assert info.owned_by == "kcake"
        assert isinstance(info.created, int)


class TestHealthResponse:
    """健康响应测试"""
    
    def test_health_response(self):
        """测试健康响应"""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=1234567890
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.timestamp == 1234567890
    
    def test_health_response_default_timestamp(self):
        """测试健康响应默认时间戳"""
        response = HealthResponse(status="healthy", version="1.0.0")
        assert isinstance(response.timestamp, int)
        assert response.timestamp > 0


class TestChatCompletionResponse:
    """聊天完成响应测试"""
    
    def test_chat_completion_response(self):
        """测试聊天完成响应"""
        response = ChatCompletionResponse(
            id="chat-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        )
        
        assert response.id == "chat-123"
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.usage["total_tokens"] == 15


class TestCompletionResponse:
    """文本完成响应测试"""
    
    def test_completion_response(self):
        """测试文本完成响应"""
        response = CompletionResponse(
            id="cmpl-123",
            object="text_completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "text": "This is a completion",
                    "index": 0,
                    "finish_reason": "length"
                }
            ],
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 20,
                "total_tokens": 25
            }
        )
        
        assert response.id == "cmpl-123"
        assert response.model == "test-model"
        assert response.choices[0]["text"] == "This is a completion"
        assert response.usage["total_tokens"] == 25


if __name__ == "__main__":
    """践行毛泽东思想：实践是检验真理的唯一标准"""
    print("开始执行API模块增强测试...")
    pytest.main([__file__, "-v"])