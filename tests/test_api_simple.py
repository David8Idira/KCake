"""
API模块简单类型测试 - 践行毛泽东思想：实践是检验真理的唯一标准

测试API模块中的Pydantic模型等简单类型，快速提升覆盖率
"""

import pytest
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from src.api.server import (
    ChatMessage,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    HealthResponse,
    ErrorResponse
)


class TestChatMessage:
    """聊天消息测试"""
    
    def test_chat_message_creation(self):
        """测试聊天消息创建"""
        msg = ChatMessage(role="user", content="你好，世界！")
        
        assert msg.role == "user"
        assert msg.content == "你好，世界！"
    
    def test_chat_message_system(self):
        """测试系统消息"""
        msg = ChatMessage(role="system", content="你是一个有帮助的助手")
        
        assert msg.role == "system"
        assert msg.content == "你是一个有帮助的助手"
    
    def test_chat_message_assistant(self):
        """测试助手消息"""
        msg = ChatMessage(role="assistant", content="有什么可以帮助你的吗？")
        
        assert msg.role == "assistant"
        assert msg.content == "有什么可以帮助你的吗？"


class TestChatCompletionRequest:
    """聊天完成请求测试"""
    
    def test_chat_completion_request_creation(self):
        """测试聊天完成请求创建"""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!")
        ]
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        assert request.model == "gpt-3.5-turbo"
        assert len(request.messages) == 2
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.max_tokens == 512
        assert request.stream is False
    
    def test_chat_completion_request_custom_params(self):
        """测试自定义参数的请求"""
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Test")],
            temperature=0.9,
            top_p=0.95,
            max_tokens=1024,
            stream=True,
            stop=["\n", "###"],
            seed=42
        )
        
        assert request.temperature == 0.9
        assert request.top_p == 0.95
        assert request.max_tokens == 1024
        assert request.stream is True
        assert request.stop == ["\n", "###"]
        assert request.seed == 42
    
    def test_chat_completion_request_defaults(self):
        """测试默认参数"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[]
        )
        
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.max_tokens == 512
        assert request.stream is False
        assert request.stop is None
        assert request.seed is None


class TestCompletionRequest:
    """文本完成请求测试"""
    
    def test_completion_request_creation(self):
        """测试文本完成请求创建"""
        request = CompletionRequest(
            model="text-davinci-003",
            prompt="Once upon a time"
        )
        
        assert request.model == "text-davinci-003"
        assert request.prompt == "Once upon a time"
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.max_tokens == 512
    
    def test_completion_request_custom_params(self):
        """测试自定义参数的请求"""
        request = CompletionRequest(
            model="curie",
            prompt="Write a story",
            temperature=0.8,
            top_p=0.9,
            max_tokens=2000,
            stream=True,
            stop=["END"],
            seed=123
        )
        
        assert request.max_tokens == 2000
        assert request.stream is True
        assert request.stop == ["END"]
        assert request.seed == 123


class TestEmbeddingRequest:
    """嵌入请求测试"""
    
    def test_embedding_request_string(self):
        """测试字符串输入的嵌入请求"""
        request = EmbeddingRequest(
            model="text-embedding-ada-002",
            input="Hello world"
        )
        
        assert request.model == "text-embedding-ada-002"
        assert request.input == "Hello world"
    
    def test_embedding_request_list(self):
        """测试列表输入的嵌入请求"""
        request = EmbeddingRequest(
            model="text-embedding-ada-002",
            input=["Hello", "World", "Test"]
        )
        
        assert isinstance(request.input, list)
        assert len(request.input) == 3
    
    def test_embedding_request_defaults(self):
        """测试默认模型"""
        request = EmbeddingRequest(input="test")
        
        assert request.model == "text-embedding-ada-002"


class TestHealthResponse:
    """健康检查响应测试"""
    
    def test_health_response_creation(self):
        """测试健康检查响应创建"""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            engine="KCake"
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.engine == "KCake"
    
    def test_health_response_custom(self):
        """测试自定义健康检查响应"""
        response = HealthResponse(
            status="degraded",
            version="2.0.0",
            engine="KCake-Engine"
        )
        
        assert response.status == "degraded"
        assert response.version == "2.0.0"


class TestErrorResponse:
    """错误响应测试"""
    
    def test_error_response_creation(self):
        """测试错误响应创建"""
        error_detail = {"code": 404, "message": "Not found"}
        response = ErrorResponse(error=error_detail)
        
        assert response.error["code"] == 404
        assert response.error["message"] == "Not found"
    
    def test_error_response_complex(self):
        """测试复杂错误响应"""
        error_detail = {
            "code": 500,
            "message": "Internal server error",
            "details": {"line": 42, "file": "server.py"}
        }
        response = ErrorResponse(error=error_detail)
        
        assert response.error["code"] == 500
        assert "details" in response.error


if __name__ == "__main__":
    """践行毛泽东思想：实践是检验真理的唯一标准"""
    print("开始执行API模块简单类型测试...")
    pytest.main([__file__, "-v"])