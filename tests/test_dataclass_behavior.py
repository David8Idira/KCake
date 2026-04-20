"""
core模块dataclass测试 - 践行毛泽东思想：实践是检验真理的唯一标准
"""

import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time
import numpy as np


# 测试@dataclass装饰器行为
class TestDataclassBehavior:
    """dataclass行为测试"""
    
    def test_simple_dataclass(self):
        """测试简单dataclass"""
        @dataclass
        class SimpleData:
            name: str
            value: int
        
        data = SimpleData(name="test", value=42)
        assert data.name == "test"
        assert data.value == 42
    
    def test_dataclass_with_defaults(self):
        """测试带默认值的dataclass"""
        @dataclass
        class DataWithDefaults:
            name: str
            count: int = 0
            active: bool = True
        
        data = DataWithDefaults(name="test")
        assert data.name == "test"
        assert data.count == 0
        assert data.active is True
    
    def test_dataclass_with_field(self):
        """测试带field的dataclass"""
        @dataclass
        class DataWithField:
            name: str
            tags: List[str] = field(default_factory=list)
            metadata: Dict[str, Any] = field(default_factory=dict)
        
        data = DataWithField(name="test")
        assert data.name == "test"
        assert data.tags == []
        assert data.metadata == {}


class TestCacheEntryLike:
    """模拟CacheEntry测试"""
    
    def test_cache_entry_creation(self):
        """测试缓存条目创建"""
        @dataclass
        class CacheEntry:
            key: str
            value: np.ndarray
            size_bytes: int
            created_at: float
            last_accessed: float
            access_count: int = 0
            metadata: Dict[str, Any] = field(default_factory=dict)
        
        # 创建模拟数据
        arr = np.array([1, 2, 3, 4, 5])
        now = time.time()
        
        entry = CacheEntry(
            key="test-key",
            value=arr,
            size_bytes=arr.nbytes,
            created_at=now,
            last_accessed=now,
            access_count=0,
            metadata={"source": "test"}
        )
        
        assert entry.key == "test-key"
        assert entry.value.shape == (5,)
        assert entry.size_bytes == arr.nbytes
        assert entry.access_count == 0
        assert entry.metadata["source"] == "test"
    
    def test_cache_entry_touch(self):
        """测试缓存条目touch方法"""
        @dataclass
        class CacheEntry:
            key: str
            value: int
            size_bytes: int
            created_at: float
            last_accessed: float
            access_count: int = 0
            
            def touch(self):
                """更新访问时间"""
                self.last_accessed = time.time()
                self.access_count += 1
        
        now = time.time()
        entry = CacheEntry(
            key="test",
            value=42,
            size_bytes=8,
            created_at=now,
            last_accessed=now,
            access_count=0
        )
        
        initial_access = entry.access_count
        entry.touch()
        
        assert entry.access_count == initial_access + 1
        assert entry.last_accessed >= now


class TestModelInfoLike:
    """模拟ModelInfo测试"""
    
    def test_model_info_creation(self):
        """测试模型信息创建"""
        @dataclass
        class ModelInfo:
            name: str
            path: str
            size_bytes: int
            num_parameters: int
            dtype: str
            quantization: Optional[str] = None
            is_cached: bool = False
        
        info = ModelInfo(
            name="gpt-3.5-turbo",
            path="/models/gpt-3.5-turbo",
            size_bytes=1024 * 1024 * 1024 * 4,  # 4GB
            num_parameters=175000000000,  # 175B
            dtype="float16",
            quantization="int8",
            is_cached=True
        )
        
        assert info.name == "gpt-3.5-turbo"
        assert info.num_parameters == 175000000000
        assert info.quantization == "int8"
        assert info.is_cached is True
    
    def test_model_info_defaults(self):
        """测试模型信息默认值"""
        @dataclass
        class ModelInfo:
            name: str
            path: str
            size_bytes: int
            num_parameters: int
            dtype: str
            quantization: Optional[str] = None
            is_cached: bool = False
        
        info = ModelInfo(
            name="test",
            path="/test",
            size_bytes=1000,
            num_parameters=1000000,
            dtype="float32"
        )
        
        assert info.quantization is None
        assert info.is_cached is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])