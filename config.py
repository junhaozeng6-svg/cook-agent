"""
RAG系统配置文件
"""

from dataclasses import asdict, dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = "./cook"
    index_save_path: str = "./vector_index"

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_provider: str = "deepseek"  # LLM提供商: moonshot / deepseek / openai
    llm_model: str = "deepseek-v4-flash-ga-260731"  # LLM模型名称: moonshot-v1 / deepseek-v4-flash / gpt-3.5-turbo / gpt-4
    llm_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"  # 火山方舟端点(ark-前缀key)
    #   deepseek 默认: https://api.deepseek.com   (模型用 deepseek-chat, 支持工具调用)
    #   openai   默认: https://api.openai.com/v1

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 1.0  # kimi-k3 仅允许 temperature=1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()