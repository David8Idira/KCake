"""
KCake 核心模块测试

测试推理引擎、模型加载器、KV缓存等功能
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.core.inference_engine import InferenceEngine, InferenceRequest, InferenceResponse
from src.core.model_loader import ModelLoader, ModelInfo
from src.core.kv_cache import KVCache, CacheEntry, MultiLevelKVCache
from src.core.token_generator import TokenGenerator, GenerationConfig, TokenizedText


class TestInferenceEngine:
    """推理引擎测试"""
    
    @pytest.fixture
    def engine(self):
        """创建推理引擎实例"""
        return InferenceEngine(device="cpu")
    
    def test_engine_initialization(self, engine):
        """测试引擎初始化"""
        assert engine.device == "cpu"
        assert engine.model is None
        assert engine.tokenizer is None
        assert engine.is_loaded is False
        assert engine.model_name is None
    
    def test_get_model_info_not_loaded(self, engine):
        """测试未加载时的模型信息"""
        info = engine.get_model_info()
        assert info["status"] == "not_loaded"
        assert info["is_loaded"] is False
    
    def test_clear_cache(self, engine):
        """测试清理缓存"""
        engine.kv_cache["test"] = {"data": "value"}
        engine.clear_cache()
        assert len(engine.kv_cache) == 0
    
    @pytest.mark.asyncio
    async def test_unload_model(self, engine):
        """测试卸载模型"""
        # 设置模拟数据
        engine.model = Mock()
        engine.tokenizer = Mock()
        engine.model_name = "test-model"
        engine.is_loaded = True
        
        # 卸载
        await engine.unload_model()
        
        assert engine.model is None
        assert engine.tokenizer is None
        assert engine.model_name is None
        assert engine.is_loaded is False


class TestInferenceRequest:
    """推理请求测试"""
    
    def test_request_creation(self):
        """测试请求创建"""
        request = InferenceRequest(
            model="test-model",
            prompt="Hello, world!",
            max_tokens=100,
            temperature=0.7
        )
        
        assert request.model == "test-model"
        assert request.prompt == "Hello, world!"
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.stream is False
    
    def test_request_defaults(self):
        """测试请求默认值"""
        request = InferenceRequest(
            model="test-model",
            prompt="Hello"
        )
        
        assert request.stream is False
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stop is None
        assert request.seed is None


class TestKVCache:
    """KV缓存测试"""
    
    @pytest.fixture
    def cache(self):
        """创建缓存实例"""
        return KVCache(max_size_mb=10, max_entries=100)
    
    def test_cache_initialization(self, cache):
        """测试缓存初始化"""
        stats = cache.get_stats()
        assert stats["current_entries"] == 0
        assert stats["current_size_mb"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
    
    def test_cache_put_and_get(self, cache):
        """测试缓存存取"""
        test_array = np.random.rand(100).astype(np.float32)
        
        cache.put("key1", test_array, {"meta": "data"})
        
        retrieved = cache.get("key1")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, test_array)
    
    def test_cache_miss(self, cache):
        """测试缓存未命中"""
        result = cache.get("nonexistent")
        assert result is None
        
        stats = cache.get_stats()
        assert stats["misses"] == 1
    
    def test_cache_hit(self, cache):
        """测试缓存命中"""
        test_array = np.random.rand(10).astype(np.float32)
        cache.put("key1", test_array)
        
        # 第一次获取
        cache.get("key1")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        
        # 第二次获取
        cache.get("key1")
        stats = cache.get_stats()
        assert stats["hits"] == 2
    
    def test_cache_delete(self, cache):
        """测试缓存删除"""
        test_array = np.random.rand(10).astype(np.float32)
        cache.put("key1", test_array)
        
        assert cache.get("key1") is not None
        
        success = cache.delete("key1")
        assert success is True
        assert cache.get("key1") is None
    
    def test_cache_clear(self, cache):
        """测试清空缓存"""
        test_array = np.random.rand(10).astype(np.float32)
        cache.put("key1", test_array)
        cache.put("key2", test_array)
        
        assert len(cache) == 2
        
        cache.clear()
        
        assert len(cache) == 0
    
    def test_cache_contains(self, cache):
        """测试包含检查"""
        test_array = np.random.rand(10).astype(np.float32)
        cache.put("key1", test_array)
        
        assert "key1" in cache
        assert "key2" not in cache
    
    def test_cache_eviction_lru(self, cache):
        """测试LRU淘汰"""
        cache.eviction_policy = "lru"
        
        # 放入多个条目
        for i in range(150):
            test_array = np.random.rand(100).astype(np.float32)
            cache.put(f"key{i}", test_array)
        
        # 检查是否淘汰了一些条目
        assert len(cache) < 150
        assert cache.get_stats()["evictions"] > 0


class TestMultiLevelKVCache:
    """多级KV缓存测试"""
    
    @pytest.fixture
    def cache(self):
        """创建多级缓存实例"""
        memory_cache = KVCache(max_size_mb=1, max_entries=10)
        return MultiLevelKVCache(memory_cache=memory_cache)
    
    def test_multilevel_initialization(self, cache):
        """测试多级缓存初始化"""
        assert cache.memory_cache is not None
        assert isinstance(cache.memory_cache, KVCache)


class TestModelLoader:
    """模型加载器测试"""
    
    @pytest.fixture
    def loader(self):
        """创建加载器实例"""
        return ModelLoader(cache_dir="/tmp/test_cache")
    
    def test_loader_initialization(self, loader):
        """测试加载器初始化"""
        assert loader.cache_dir == "/tmp/test_cache"
        assert len(loader.loaded_models) == 0
    
    def test_get_cached_models(self, loader):
        """测试获取缓存模型"""
        models = loader.get_cached_models()
        assert isinstance(models, dict)
        assert len(models) == 0
    
    def test_get_cache_size(self, loader):
        """测试获取缓存大小"""
        size = loader.get_cache_size()
        assert size == 0


class TestTokenGenerator:
    """Token生成器测试"""
    
    @pytest.fixture
    def generator(self):
        """创建生成器实例"""
        return TokenGenerator(max_length=2048)
    
    def test_generator_initialization(self, generator):
        """测试生成器初始化"""
        assert generator.max_length == 2048
        assert generator.tokenizer is None
    
    def test_set_tokenizer(self, generator):
        """测试设置分词器"""
        mock_tokenizer = Mock()
        generator.set_tokenizer(mock_tokenizer)
        assert generator.tokenizer == mock_tokenizer
    
    def test_count_tokens_without_tokenizer(self, generator):
        """测试无分词器时计数"""
        with pytest.raises(RuntimeError):
            generator.count_tokens("Hello")


class TestGenerationConfig:
    """生成配置测试"""
    
    def test_config_defaults(self):
        """测试配置默认值"""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 512
        assert config.min_new_tokens == 1
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.0
        assert config.num_beams == 1
        assert config.do_sample is True
    
    def test_config_custom(self):
        """测试自定义配置"""
        config = GenerationConfig(
            max_new_tokens=1000,
            temperature=0.5,
            top_p=0.8
        )
        
        assert config.max_new_tokens == 1000
        assert config.temperature == 0.5
        assert config.top_p == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])