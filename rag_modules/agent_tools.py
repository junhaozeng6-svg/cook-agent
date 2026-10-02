"""
Agent工具层 - 将现有RAG检索/数据模块封装为Agent可调用的Function Tools

设计思路:
    现有模块(retrieval_optimization / data_preparation)一行不改,
    通过 make_tools 工厂函数把它们的函数包装成 LangChain @tool,
    LLM 通过工具描述自主决定"调哪个工具、传什么参数、何时停止"。
"""

import logging
from typing import List, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def make_tools(retrieval_module, data_module, generation_module=None,
               top_k: int = 3, include: Optional[List[str]] = None) -> List:
    """
    基于现有RAG模块实例创建Agent工具列表。

    Args:
        retrieval_module: RetrievalOptimizationModule 实例(提供混合检索)
        data_module:      DataPreparationModule 实例(提供文档/父文档/过滤)
        generation_module: GenerationIntegrationModule 实例(提供查询重写, 可选)
        top_k:            检索返回的候选数量
        include:          需要返回的工具名子集(如 ['search_recipe', 'get_full_recipe']);
                          为 None 时返回全部工具(用于多智能体场景按角色裁剪工具集)

    Returns:
        List[BaseTool]: 工具列表(search_recipe / get_full_recipe / filter_by_category /
                               list_categories / rewrite_query)
    """
    supported_categories = data_module.get_supported_categories()
    supported_difficulties = data_module.get_supported_difficulties()

    @tool
    def search_recipe(query: str) -> str:
        """按语义与关键词混合检索菜谱内容块, 返回命中的菜谱章节文本(含菜名/分类/难度)。
        适用于查找与某个菜品、食材或做法相关的菜谱信息。
        query 应包含具体的菜品名、食材或做法关键词, 如"红烧肉""清蒸鲈鱼的做法"。"""
        chunks = retrieval_module.hybrid_search(query, top_k=top_k)
        if not chunks:
            return "未找到相关菜谱。"
        lines = []
        for c in chunks:
            dish = c.metadata.get('dish_name', '未知菜品')
            cat = c.metadata.get('category', '')
            diff = c.metadata.get('difficulty', '')
            head = f"【{dish}】分类:{cat} 难度:{diff}"
            content = c.page_content.strip().replace('\n', ' ')
            lines.append(f"{head}\n{content[:500]}")
        return "\n\n".join(lines)

    @tool
    def get_full_recipe(query: str) -> str:
        """检索并返回完整菜谱(整篇原始内容, 含必备原料/计算/操作/附加内容)。
        适用于回答"怎么做/需要什么食材/制作步骤"这类需要完整上下文的问题。
        query 可包含菜品名或做法关键词。"""
        chunks = retrieval_module.hybrid_search(query, top_k=top_k)
        if not chunks:
            return "未找到相关菜谱。"
        docs = data_module.get_parent_documents(chunks)
        parts = []
        for d in docs:
            dish = d.metadata.get('dish_name', '未知菜品')
            parts.append(f"【完整菜谱: {dish}】\n{d.page_content}")
        return "\n\n".join(parts)

    @tool
    def filter_by_category(category: Optional[str] = None,
                           difficulty: Optional[str] = None) -> str:
        """按菜品分类和/或难度过滤菜谱, 返回符合条件的菜名列表。
        适用于"推荐/有哪些"类查询(如"推荐几个素菜""简单的汤")。
        category 可选值: {', '.join(supported_categories)}。
        difficulty 可选值: {', '.join(supported_difficulties)}。"""
        docs = data_module.documents
        if category:
            docs = [d for d in docs if d.metadata.get('category') == category]
        if difficulty:
            docs = [d for d in docs if d.metadata.get('difficulty') == difficulty]
        if not docs:
            return "没有符合条件的菜谱。"
        names = [d.metadata.get('dish_name', '未知') for d in docs]
        return "符合条件菜品:\n" + "\n".join(f"- {n}" for n in names[:20])

    @tool
    def list_categories() -> str:
        """列出知识库支持的所有菜品分类, 供用户了解可查询的类别。"""
        return "可用分类: " + ", ".join(supported_categories)

    tools_list = [search_recipe, get_full_recipe, filter_by_category, list_categories]

    # 自纠错工具: 检索无果时重写查询再查(Agentic RAG 闭环)
    if generation_module is not None:
        @tool
        def rewrite_query(query: str) -> str:
            """重写查询以改进检索效果。当 search_recipe 或 get_full_recipe 返回"未找到"或结果不足时调用,
            让后续检索命中更相关的菜谱。query 为原始查询, 返回重写后的查询文本。"""
            rewritten = generation_module.query_rewrite(query)
            return f"重写后的查询: {rewritten}"

        tools_list.append(rewrite_query)

    # 按需裁剪工具子集(多智能体场景: 每个Worker只暴露自己的工具)
    if include is not None:
        tools_list = [t for t in tools_list if t.name in include]

    return tools_list
