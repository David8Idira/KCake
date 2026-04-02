"""
KCake Quantization - 量化配置

支持 INT4, INT8, FP8 量化
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    """量化配置"""
    quant_type: str  # int4, int8, fp8
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


def get_quantization_config(quant_type: str) -> Optional[QuantizationConfig]:
    """
    获取量化配置
    
    Args:
        quant_type: 量化类型 (int4, int8, fp8, None)
        
    Returns:
        量化配置对象，如果 quant_type 为 None 则返回 None
    """
    if quant_type is None:
        return None
    
    quant_type = quant_type.lower()
    
    if quant_type == "int4":
        return QuantizationConfig(
            quant_type="int4",
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quant_type == "int8":
        return QuantizationConfig(
            quant_type="int8",
            load_in_4bit=False,
            load_in_8bit=True,
        )
    elif quant_type == "fp8":
        # FP8 需要特定硬件支持
        return QuantizationConfig(
            quant_type="fp8",
            load_in_4bit=False,
            load_in_8bit=False,
        )
    else:
        raise ValueError(f"Unknown quantization type: {quant_type}")


def get_bnb_config(config: QuantizationConfig) -> Optional[Dict[str, Any]]:
    """
    获取 bitsandbytes 配置
    """
    if config.quant_type == "int4":
        from bitsandbytes import BitsAndBytesConfig
        
        return BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
    elif config.quant_type == "int8":
        from bitsandbytes import BitsAndBytesConfig
        
        return BitsAndBytesConfig(
            load_in_8bit=config.load_in_8bit,
        )
    
    return None
