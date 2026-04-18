"""
KCake API测试

测试API服务和OpenAI/Ollama适配器
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.api.server import KcakeAPIServer, ChatCompletionRequest, CompletionRequest
from src.api.openai_adapter import OpenAIAdapter
from src.api.ollama_adapter import OllamaAdapter


class TestChatCompletionRequest:
    """聊天完成请求测试"""
    
    def test_request_creation(self):
        """测试请求创建"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100
        )
        
        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.max_tokens == 100
    
    def test_request_defaults(self):
        """测试请求默认值"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.max_tokens == 512
        assert request.stream is False


class TestCompletionRequest:
    """文本完成请求测试"""
    
    def test_request_creation(self):
        """测试请求创建"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello, world!",
            max_tokens=100
        )
        
        assert request.model == "test-model"
        assert request.prompt == "Hello, world!"
        assert request.max_tokens == 100
    
    def test_request_defaults(self):
        """测试请求默认值"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello"
        )
        
        assert request.temperature == 0.7
        assert request.max_tokens == 512


class TestOpenAIAdapter:
    """OpenAI适配器测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """创建模拟推理引擎"""
        engine = Mock()
        engine.generate = AsyncMock()
        engine.get_model_info = Mock(return_value={"model_name": "test-model"})
        return engine
    
    @pytest.fixture
    def adapter(self, mock_engine):
        """创建适配器实例"""
        return OpenAIAdapter(mock_engine)
    
    def test_adapter_initialization(self, adapter):
        """测试适配器初始化"""
        assert adapter.api_version == "v1"
    
    def test_list_models(self, adapter):
        """测试获取模型列表"""
        result = adapter.list_models()
        assert "object" in result
        assert result["object"] == "list"
        assert "data" in result


class TestOllamaAdapter:
    """Ollama适配器测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """创建模拟推理引擎"""
        engine = Mock()
        engine.generate = AsyncMock()
        engine.get_model_info = Mock(return_value={"model_name": "test-model"})
        return engine
    
    @pytest.fixture
    def adapter(self, mock_engine):
        """创建适配器实例"""
        return OllamaAdapter(mock_engine)
    
    def test_adapter_initialization(self, adapter):
        """测试适配器初始化"""
        assert adapter is not None
    
    @pytest.mark.asyncio
    async def test_tags(self, adapter):
        """测试获取模型标签"""
        result = await adapter.tags()
        assert "models" in result
        assert isinstance(result["models"], list)


class TestKcakeAPIServer:
    """KCake API服务器测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """创建模拟推理引擎"""
        engine = Mock()
        engine.get_model_info = Mock(return_value={
            "is_loaded": False,
            "model_name": "none"
        })
        return engine
    
    def test_server_initialization(self, mock_engine):
        """测试服务器初始化"""
        server = KcakeAPIServer(
            inference_engine=mock_engine,
            host="127.0.0.1",
            port=8000
        )
        
        assert server.host == "127.0.0.1"
        assert server.port == 8000
        assert server.app is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])