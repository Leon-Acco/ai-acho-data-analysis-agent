"""
多模型配置模块

支持智谱(ChatGLM)、DeepSeek、豆包(Doubao)、千问(Qwen)等大模型
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ModelProvider(Enum):
    """模型提供商枚举"""
    DEEPSEEK = "deepseek"
    ZHIPU = "zhipu"  # 智谱
    DOUBAO = "doubao"  # 豆包
    QWEN = "qwen"  # 千问
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelConfig:
    """模型配置"""
    provider: ModelProvider
    model_name: str
    api_base: str
    display_name: str

    def __str__(self) -> str:
        return f"{self.display_name} ({self.provider.value})"


# 预设模型配置
AVAILABLE_MODELS: dict[str, ModelConfig] = {
    # DeepSeek 系列
    "deepseek-chat": ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-chat",
        api_base="https://api.deepseek.com",
        display_name="DeepSeek Chat"
    ),
    "deepseek-reasoner": ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-reasoner",
        api_base="https://api.deepseek.com",
        display_name="DeepSeek Reasoner"
    ),
    
    # 智谱(ChatGLM) 系列
    "glm-4": ModelConfig(
        provider=ModelProvider.ZHIPU,
        model_name="glm-4",
        api_base="https://open.bigmodel.cn/api/paas/v4",
        display_name="智谱 GLM-4"
    ),
    "glm-4-plus": ModelConfig(
        provider=ModelProvider.ZHIPU,
        model_name="glm-4-plus",
        api_base="https://open.bigmodel.cn/api/paas/v4",
        display_name="智谱 GLM-4-Plus"
    ),
    "glm-3-turbo": ModelConfig(
        provider=ModelProvider.ZHIPU,
        model_name="glm-3-turbo",
        api_base="https://open.bigmodel.cn/api/paas/v4",
        display_name="智谱 GLM-3-Turbo"
    ),
    
    # 豆包(Doubao) 系列
    "doubao-pro-32k": ModelConfig(
        provider=ModelProvider.DOUBAO,
        model_name="doubao-pro-32k",
        api_base="https://ark.cn-beijing.volces.com/api/v3",
        display_name="豆包 Doubao Pro 32K"
    ),
    "doubao-pro-128k": ModelConfig(
        provider=ModelProvider.DOUBAO,
        model_name="doubao-pro-128k",
        api_base="https://ark.cn-beijing.volces.com/api/v3",
        display_name="豆包 Doubao Pro 128K"
    ),
    
    # 千问(Qwen) 系列
    "qwen-turbo": ModelConfig(
        provider=ModelProvider.QWEN,
        model_name="qwen-turbo",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        display_name="通义千问 Turbo"
    ),
    "qwen-plus": ModelConfig(
        provider=ModelProvider.QWEN,
        model_name="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        display_name="通义千问 Plus"
    ),
    "qwen-max": ModelConfig(
        provider=ModelProvider.QWEN,
        model_name="qwen-max",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        display_name="通义千问 Max"
    ),
    
    # OpenAI 兼容
    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o",
        api_base="https://api.openai.com/v1",
        display_name="OpenAI GPT-4o"
    ),
    "gpt-4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        api_base="https://api.openai.com/v1",
        display_name="OpenAI GPT-4o Mini"
    ),
}


def get_model_config(model_key: str) -> Optional[ModelConfig]:
    """获取模型配置"""
    return AVAILABLE_MODELS.get(model_key)


def get_all_models() -> dict[str, ModelConfig]:
    """获取所有可用模型"""
    return AVAILABLE_MODELS.copy()


def get_models_by_provider(provider: ModelProvider) -> dict[str, ModelConfig]:
    """获取指定提供商的模型"""
    return {
        key: config for key, config in AVAILABLE_MODELS.items()
        if config.provider == provider
    }


def get_default_model() -> str:
    """获取默认模型"""
    return "deepseek-chat"


def get_model_env_var(provider: ModelProvider) -> str:
    """获取模型提供商对应的环境变量名"""
    env_map = {
        ModelProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
        ModelProvider.ZHIPU: "ZHIPU_API_KEY",
        ModelProvider.DOUBAO: "DOUBAO_API_KEY",
        ModelProvider.QWEN: "QWEN_API_KEY",
        ModelProvider.OPENAI: "OPENAI_API_KEY",
        ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    }
    return env_map.get(provider, f"{provider.value.upper()}_API_KEY")


# 推荐的免费/低成本模型
RECOMMENDED_MODELS = [
    "deepseek-chat",
    "glm-3-turbo",
    "doubao-pro-32k",
    "qwen-turbo",
]
