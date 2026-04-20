"""
量化模块测试 - 践行毛泽东思想：实践是检验真理的唯一标准

测试量化配置和功能
"""

import pytest
from src.quantization import QuantizationConfig, get_quantization_config


class TestQuantizationConfig:
    """量化配置测试"""
    
    def test_config_creation(self):
        """测试配置创建"""
        config = QuantizationConfig(
            quant_type="int4",
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16"
        )
        
        assert config.quant_type == "int4"
        assert config.load_in_4bit is True
        assert config.bnb_4bit_compute_dtype == "float16"
    
    def test_config_defaults(self):
        """测试配置默认值"""
        config = QuantizationConfig(quant_type="int8")
        
        assert config.quant_type == "int8"
        assert config.load_in_4bit is False
        assert config.load_in_8bit is False
        assert config.bnb_4bit_use_double_quant is True
        assert config.bnb_4bit_quant_type == "nf4"
    
    def test_config_with_all_fields(self):
        """测试包含所有字段的配置"""
        config = QuantizationConfig(
            quant_type="fp8",
            load_in_4bit=False,
            load_in_8bit=False,
            bnb_4bit_compute_dtype="float32",
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False
        )
        
        assert config.quant_type == "fp8"
        assert config.load_in_4bit is False
        assert config.load_in_8bit is False
        assert config.bnb_4bit_compute_dtype == "float32"
        assert config.bnb_4bit_quant_type == "fp4"
        assert config.bnb_4bit_use_double_quant is False
    
    def test_config_repr(self):
        """测试配置的字符串表示"""
        config = QuantizationConfig(quant_type="int4")
        repr_str = repr(config)
        
        assert "QuantizationConfig" in repr_str
        assert "quant_type='int4'" in repr_str


class TestGetQuantizationConfig:
    """获取量化配置测试"""
    
    def test_get_int4_config(self):
        """测试获取INT4配置"""
        config = get_quantization_config("int4")
        
        assert config is not None
        assert config.quant_type == "int4"
        assert config.load_in_4bit is True
        assert config.load_in_8bit is False
        assert config.bnb_4bit_compute_dtype == "float16"
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True
    
    def test_get_int8_config(self):
        """测试获取INT8配置"""
        config = get_quantization_config("int8")
        
        assert config is not None
        assert config.quant_type == "int8"
        assert config.load_in_4bit is False
        assert config.load_in_8bit is True
    
    def test_get_fp8_config(self):
        """测试获取FP8配置"""
        config = get_quantization_config("fp8")
        
        assert config is not None
        assert config.quant_type == "fp8"
        assert config.load_in_4bit is False
        assert config.load_in_8bit is False
    
    def test_get_none_config(self):
        """测试获取None配置"""
        config = get_quantization_config(None)
        
        assert config is None
    
    def test_get_invalid_config(self):
        """测试获取无效配置 - 应该抛出ValueError"""
        with pytest.raises(ValueError, match="Unknown quantization type"):
            get_quantization_config("invalid")
    
    def test_case_insensitive(self):
        """测试大小写不敏感"""
        config1 = get_quantization_config("INT4")
        config2 = get_quantization_config("int4")
        
        assert config1 is not None
        assert config2 is not None
        assert config1.quant_type == config2.quant_type


if __name__ == "__main__":
    """践行毛泽东思想：实践是检验真理的唯一标准"""
    print("开始执行量化模块测试...")
    pytest.main([__file__, "-v"])