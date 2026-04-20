"""
核心模块简单类型测试 - 践行毛泽东思想：实践是检验真理的唯一标准

测试core模块中的dataclass等简单类型，快速提升覆盖率
"""

import pytest
from typing import Dict, List
from src.core.inference_engine import (
    InferenceRequest,
    InferenceResponse
)
from src.core.token_generator import (
    GenerationConfig,
    TokenizedText
)
from src.core.model_loader import ModelInfo
from src.core.kv_cache import CacheEntry


class TestInferenceRequest:
    """推理请求测试"""
    
    def test_inference_request_creation(self):
        """测试推理请求创建"""
        request = InferenceRequest(
            model="gpt-3.5-turbo",
            prompt="你好，世界！",
            stream=False,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        
        assert request.model == "gpt-3.5-turbo"
        assert request.prompt == "你好，世界！"
        assert request.stream is False
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stop is None
        assert request.seed is None
    
    def test_inference_request_with_options(self):
        """测试带选项的推理请求"""
        request = InferenceRequest(
            model="deepseek-v3",
            prompt="What is the meaning of life?",
            stream=True,
            max_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            stop=["\n", "###"],
            seed=42
        )
        
        assert request.model == "deepseek-v3"
        assert request.stream is True
        assert request.max_tokens == 1024
        assert request.temperature == 0.8
        assert request.top_p == 0.95
        assert request.stop == ["\n", "###"]
        assert request.seed == 42
    
    def test_inference_request_repr(self):
        """测试推理请求字符串表示"""
        request = InferenceRequest(
            model="test-model",
            prompt="test prompt"
        )
        
        repr_str = repr(request)
        assert "InferenceRequest" in repr_str
        assert "test-model" in repr_str


class TestInferenceResponse:
    """推理响应测试"""
    
    def test_inference_response_creation(self):
        """测试推理响应创建"""
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        
        response = InferenceResponse(
            text="这是一个测试响应。",
            tokens_generated=20,
            finish_reason="stop",
            model="gpt-3.5-turbo",
            usage=usage
        )
        
        assert response.text == "这是一个测试响应。"
        assert response.tokens_generated == 20
        assert response.finish_reason == "stop"
        assert response.model == "gpt-3.5-turbo"
        assert response.usage == usage
    
    def test_inference_response_with_length_reason(self):
        """测试长度限制完成的推理响应"""
        usage = {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}
        
        response = InferenceResponse(
            text="This response was cut short due to length.",
            tokens_generated=100,
            finish_reason="length",
            model="claude-3",
            usage=usage
        )
        
        assert response.finish_reason == "length"
        assert response.tokens_generated == 100
        assert response.usage["total_tokens"] == 150
    
    def test_inference_response_repr(self):
        """测试推理响应字符串表示"""
        usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        
        response = InferenceResponse(
            text="test",
            tokens_generated=2,
            finish_reason="stop",
            model="test",
            usage=usage
        )
        
        repr_str = repr(response)
        assert "InferenceResponse" in repr_str
        assert "tokens_generated=2" in repr_str


class TestGenerationConfig:
    """生成配置测试"""
    
    def test_generation_config_defaults(self):
        """测试生成配置默认值"""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 512
        assert config.min_new_tokens == 1
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.0
        assert config.length_penalty == 1.0
        assert config.num_beams == 1
        assert config.early_stopping is False
        assert config.do_sample is True
        assert config.pad_token_id is None
        assert config.eos_token_id is None
    
    def test_generation_config_custom(self):
        """测试自定义生成配置"""
        config = GenerationConfig(
            max_new_tokens=1024,
            min_new_tokens=10,
            temperature=1.0,
            top_p=0.95,
            top_k=100,
            repetition_penalty=1.2,
            length_penalty=0.8,
            num_beams=4,
            early_stopping=True,
            do_sample=False,
            pad_token_id=0,
            eos_token_id=2
        )
        
        assert config.max_new_tokens == 1024
        assert config.min_new_tokens == 10
        assert config.temperature == 1.0
        assert config.top_p == 0.95
        assert config.top_k == 100
        assert config.repetition_penalty == 1.2
        assert config.length_penalty == 0.8
        assert config.num_beams == 4
        assert config.early_stopping is True
        assert config.do_sample is False
        assert config.pad_token_id == 0
        assert config.eos_token_id == 2
    
    def test_generation_config_repr(self):
        """测试生成配置字符串表示"""
        config = GenerationConfig()
        
        repr_str = repr(config)
        assert "GenerationConfig" in repr_str
        assert "max_new_tokens=512" in repr_str


class TestTokenizedText:
    """Token化文本测试"""
    
    def test_tokenized_text_creation(self):
        """测试Token化文本创建"""
        tokenized = TokenizedText(
            input_ids=[101, 202, 303, 404],
            attention_mask=[1, 1, 1, 1],
            text="Hello world",
            num_tokens=4
        )
        
        assert tokenized.input_ids == [101, 202, 303, 404]
        assert tokenized.attention_mask == [1, 1, 1, 1]
        assert tokenized.text == "Hello world"
        assert tokenized.num_tokens == 4
    
    def test_tokenized_text_chinese(self):
        """测试中文Token化文本"""
        tokenized = TokenizedText(
            input_ids=[1, 2, 3, 4, 5],
            attention_mask=[1, 1, 1, 1, 1],
            text="你好，世界！",
            num_tokens=5
        )
        
        assert tokenized.text == "你好，世界！"
        assert tokenized.num_tokens == 5
        assert len(tokenized.input_ids) == 5
    
    def test_tokenized_text_repr(self):
        """测试Token化文本字符串表示"""
        tokenized = TokenizedText(
            input_ids=[1, 2, 3],
            attention_mask=[1, 1, 1],
            text="test",
            num_tokens=3
        )
        
        repr_str = repr(tokenized)
        assert "TokenizedText" in repr_str
        assert "num_tokens=3" in repr_str


if __name__ == "__main__":
    """践行毛泽东思想：实践是检验真理的唯一标准"""
    print("开始执行核心模块简单类型测试...")
    pytest.main([__file__, "-v"])