"""
KCake API 服务模块

提供OpenAI兼容API、Ollama兼容API、REST API等接口服务
"""

__version__ = "0.1.0"
__author__ = "KCake Team"

from .server import KcakeAPIServer
from .openai_adapter import OpenAIAdapter
from .ollama_adapter import OllamaAdapter
from .rest_routes import setup_routes

__all__ = [
    "KcakeAPIServer",
    "OpenAIAdapter",
    "OllamaAdapter",
    "setup_routes",
]