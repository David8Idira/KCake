"""
Token生成器模块

负责文本的tokenization和解码
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """生成配置"""
    max_new_tokens: int = 512
    min_new_tokens: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    early_stopping: bool = False
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class TokenizedText:
    """Token化后的文本"""
    input_ids: List[int]
    attention_mask: List[int]
    text: str
    num_tokens: int


class TokenGenerator:
    """Token生成器"""
    
    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        max_length: int = 2048
    ):
        """
        初始化Token生成器
        
        Args:
            tokenizer: 分词器实例
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Token生成器初始化: max_length={max_length}")
    
    def set_tokenizer(self, tokenizer: Any) -> None:
        """
        设置分词器
        
        Args:
            tokenizer: 分词器实例
        """
        self.tokenizer = tokenizer
        logger.info("分词器已设置")
    
    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = True,
        max_length: Optional[int] = None
    ) -> TokenizedText:
        """
        Token化文本
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊token
            padding: 是否填充
            truncation: 是否截断
            max_length: 最大长度
            
        Returns:
            TokenizedText: Token化结果
        """
        if self.tokenizer is None:
            raise RuntimeError("分词器未设置")
        
        # Token化
        result = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length or self.max_length,
            return_tensors="np"
        )
        
        input_ids = result["input_ids"].tolist()[0] if hasattr(result["input_ids"], "tolist") else result["input_ids"][0]
        attention_mask = result["attention_mask"].tolist()[0] if hasattr(result["attention_mask"], "tolist") else result["attention_mask"][0]
        
        return TokenizedText(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text=text,
            num_tokens=len(input_ids)
        )
    
    def batch_tokenize(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None
    ) -> List[TokenizedText]:
        """
        批量Token化
        
        Args:
            texts: 文本列表
            add_special_tokens: 是否添加特殊token
            padding: 是否填充
            truncation: 是否截断
            max_length: 最大长度
            
        Returns:
            List[TokenizedText]: Token化结果列表
        """
        if self.tokenizer is None:
            raise RuntimeError("分词器未设置")
        
        # 批量Token化
        results = self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length or self.max_length,
            return_tensors="np"
        )
        
        tokenized_texts = []
        for i in range(len(texts)):
            input_ids = results["input_ids"][i].tolist() if hasattr(results["input_ids"], "tolist") else results["input_ids"][i]
            attention_mask = results["attention_mask"][i].tolist() if hasattr(results["attention_mask"], "tolist") else results["attention_mask"][i]
            
            tokenized_texts.append(TokenizedText(
                input_ids=input_ids,
                attention_mask=attention_mask,
                text=texts[i],
                num_tokens=len(input_ids)
            ))
        
        return tokenized_texts
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """
        解码token序列
        
        Args:
            token_ids: token ID序列
            skip_special_tokens: 是否跳过特殊token
            clean_up_tokenization_spaces: 是否清理tokenization空格
            
        Returns:
            str: 解码后的文本
        """
        if self.tokenizer is None:
            raise RuntimeError("分词器未设置")
        
        # 解码
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        
        return text
    
    def decode_stream(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ):
        """
        流式解码
        
        Args:
            token_ids: token ID序列
            skip_special_tokens: 是否跳过特殊token
            
        Yields:
            str: 解码后的文本片段
        """
        if self.tokenizer is None:
            raise RuntimeError("分词器未设置")
        
        # 获取特殊token
        bos_token = self.tokenizer.bos_token if hasattr(self.tokenizer, "bos_token") else None
        eos_token = self.tokenizer.eos_token if hasattr(self.tokenizer, "eos_token") else None
        
        current_text = ""
        for i, token_id in enumerate(token_ids):
            # 解码当前token
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=skip_special_tokens)
            
            # 跳过特殊token
            if skip_special_tokens:
                if bos_token and token_text == bos_token:
                    continue
                if eos_token and token_text == eos_token:
                    break
            
            current_text += token_text
            yield token_text
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数
        
        Args:
            text: 输入文本
            
        Returns:
            int: token数量
        """
        if self.tokenizer is None:
            raise RuntimeError("分词器未设置")
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def truncate_tokens(
        self,
        token_ids: List[int],
        max_length: int
    ) -> List[int]:
        """
        截断token序列
        
        Args:
            token_ids: token ID序列
            max_length: 最大长度
            
        Returns:
            List[int]: 截断后的序列
        """
        if len(token_ids) <= max_length:
            return token_ids
        
        return token_ids[:max_length]
    
    def pad_sequence(
        self,
        token_ids: List[int],
        max_length: int,
        pad_token_id: int = 0
    ) -> List[int]:
        """
        填充序列
        
        Args:
            token_ids: token ID序列
            max_length: 目标长度
            pad_token_id: 填充token的ID
            
        Returns:
            List[int]: 填充后的序列
        """
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        
        padding_length = max_length - len(token_ids)
        return token_ids + [pad_token_id] * padding_length
    
    def get_vocab_size(self) -> int:
        """获取词表大小"""
        if self.tokenizer is None:
            raise RuntimeError("分词器未设置")
        
        return len(self.tokenizer)
    
    def get_special_tokens(self) -> Dict[str, int]:
        """获取特殊token映射"""
        if self.tokenizer is None:
            raise RuntimeError("分词器未设置")
        
        special_tokens = {}
        
        if hasattr(self.tokenizer, "bos_token") and self.tokenizer.bos_token:
            special_tokens["bos"] = self.tokenizer.bos_token_id
        if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token:
            special_tokens["eos"] = self.tokenizer.eos_token_id
        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token:
            special_tokens["pad"] = self.tokenizer.pad_token_id
        if hasattr(self.tokenizer, "unk_token") and self.tokenizer.unk_token:
            special_tokens["unk"] = self.tokenizer.unk_token_id
        
        return special_tokens


class StreamingGenerator:
    """流式生成器"""
    
    def __init__(
        self,
        model: Any,
        token_generator: TokenGenerator,
        device: str = "cpu"
    ):
        """
        初始化流式生成器
        
        Args:
            model: 语言模型
            token_generator: Token生成器
            device: 设备类型
        """
        self.model = model
        self.token_generator = token_generator
        self.device = device
        
        logger.info(f"流式生成器初始化: device={device}")
    
    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig
    ):
        """
        流式生成
        
        Args:
            prompt: 输入提示
            config: 生成配置
            
        Yields:
            str: 生成的文本片段
        """
        # Token化输入
        tokenized = self.token_generator.tokenize(prompt)
        input_ids = tokenized.input_ids
        
        # 转换为tensor
        import torch
        input_ids_tensor = torch.tensor([input_ids]).to(self.device)
        
        # 逐步生成
        generated_ids = []
        past_key_values = None
        
        for _ in range(config.max_new_tokens):
            # 获取模型输出
            with torch.no_grad():
                if past_key_values is None:
                    outputs = self.model(input_ids_tensor)
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                else:
                    outputs = self.model(
                        input_ids_tensor[:, -1:],
                        past_key_values=past_key_values
                    )
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
            
            # 应用温度
            if config.do_sample and config.temperature > 0:
                logits = logits / config.temperature
            
            # 应用top-k
            if config.top_k > 0:
                indices_to_remove = torch.arange(logits.shape[-1])[None] >= config.top_k
                logits[indices_to_remove] = float('-inf')
            
            # 应用top-p
            if config.do_sample and config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # 应用repetition penalty
            if config.repetition_penalty != 1.0 and generated_ids:
                for token_id in set(generated_ids):
                    logits[0, token_id] /= config.repetition_penalty
            
            # 采样
            if config.do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)[0, 0].item()
            else:
                next_token_id = torch.argmax(logits[0]).item()
            
            # 检查是否停止
            if (
                next_token_id == config.eos_token_id
                or (config.early_stopping and len(generated_ids) >= config.min_new_tokens)
            ):
                break
            
            # 保存生成的token
            generated_ids.append(next_token_id)
            
            # 解码并yield
            token_text = self.token_generator.decode([next_token_id])
            yield token_text
            
            # 更新input_ids
            input_ids_tensor = torch.tensor([[next_token_id]]).to(self.device)
        
        logger.info(f"流式生成完成，共生成 {len(generated_ids)} 个token")