"""
Agent编排层 - LangGraph ReAct 循环

状态图结构:
    START → agent(LLM决策) ⇄ tools(执行工具) → generator(生成/收尾) → END

循环条件(三个闸门):
    1. LLM 输出 final answer(无 tool_calls)  → 进 generator 收尾
    2. 步数 step ≥ max_steps                 → 强制进 generator, 防死循环
    3. 工具返回空结果                         → 由 LLM 自行决定重试(换关键词)或如实告知
"""

import logging
from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 3
DEFAULT_MAX_REFLECTIONS = 1

# Reflexion 反思提示词: 生成后自我批评, 不合格则按意见修订
REFLEXION_PROMPT = """你是回答质量检查员。判断下面的回答是否合格, 只输出 "PASS" 或 "REVISE: <具体改进建议>"。

判断标准:
1. 回答忠实于菜谱原文, 没有编造不存在的信息
2. 完整回答了用户问题
3. 结构清晰、实用

用户问题: {question}

回答:
{answer}

检查结果:"""

REVISE_PROMPT = """根据反思意见改进回答。严格保持忠实于食谱原文, 不得编造。

用户问题: {question}

相关食谱信息:
{context}

反思意见: {critique}

改进后的回答:"""


class AgentState(TypedDict):
    """LangGraph 状态: 对话消息 + 循环步数 + 反思次数 + 最终答案"""
    messages: Annotated[list, add_messages]
    step: int
    reflections: int
    answer: str


def build_agent_graph(retrieval_module, data_module, generation_module,
                      top_k: int = 3, max_steps: int = DEFAULT_MAX_STEPS,
                      max_reflections: int = DEFAULT_MAX_REFLECTIONS,
                      tool_names: Optional[List[str]] = None):
    """
    构建 ReAct 循环状态图: agent ⇄ tools → generator(含 Reflexion 反思)

    Args:
        retrieval_module:   RetrievalOptimizationModule 实例
        data_module:        DataPreparationModule 实例
        generation_module:  GenerationIntegrationModule 实例(提供 llm 与生成函数)
        top_k:              检索候选数量
        max_steps:          循环最大步数(防死循环)
        max_reflections:    生成后自我反思的最大次数(Reflexion)
        tool_names:         工具子集(多智能体场景每个Worker只绑定自己的工具);
                            None 表示使用全部工具

    Returns:
        编译好的 CompiledStateGraph
    """
    from .agent_tools import make_tools

    tools = make_tools(retrieval_module, data_module, generation_module,
                       top_k=top_k, include=tool_names)
    tool_map = {t.name: t for t in tools}
    llm_with_tools = generation_module.llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """LLM 决策节点: 输出 (思考 + 行动) 或 (思考 + 最终回答)"""
        response = llm_with_tools.invoke(state["messages"])
        return {
            "messages": [response],
            "step": state.get("step", 0) + 1,
        }

    def tools_node(state: AgentState) -> dict:
        """工具执行节点: 逐个执行 LLM 请求的工具调用, 结果作为 ToolMessage 回填"""
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": []}

        outputs = []
        for tc in last.tool_calls:
            name = tc.get("name")
            args = tc.get("args", {})
            try:
                result = tool_map[name].invoke(args)
            except Exception as e:
                logger.warning(f"工具 {name} 执行失败: {e}")
                result = f"工具 {name} 执行出错: {e}"
            outputs.append(ToolMessage(
                content=str(result),
                tool_call_id=tc.get("id", ""),
                name=name,
            ))
        return {"messages": outputs}

    def should_continue(state: AgentState) -> str:
        """循环条件: 还有工具调用且未超步数 → 继续; 否则收尾"""
        if state.get("step", 0) >= max_steps:
            logger.info(f"达到最大步数 {max_steps}, 强制收尾")
            return "end"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    def generator_node(state: AgentState) -> dict:
        """生成收尾节点: 有检索上下文则复用现有生成模块, 否则采用 Agent 最终文本"""
        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        question = user_msgs[-1].content if user_msgs else ""

        # 收集有效的检索上下文(工具返回了真实菜谱内容)
        docs = [
            Document(page_content=m.content, metadata={})
            for m in tool_msgs
            if m.content
            and "未找到" not in m.content
            and "没有符合条件" not in m.content
            and "执行出错" not in m.content
        ]

        if not docs:
            # 无检索上下文: 直接采用 Agent 给出的最终回答文本
            final_ai = [
                m for m in state["messages"]
                if isinstance(m, AIMessage) and m.content and not m.tool_calls
            ]
            if final_ai:
                return {"answer": final_ai[-1].content}
            return {"answer": "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"}

        # 复用现有生成模块的基础回答模式(含 _build_context 格式化)
        answer = generation_module.generate_basic_answer(question, docs)

        # Reflexion: 生成后自我反思, 判定不合格则按反思意见修订一次
        reflections = state.get("reflections", 0)
        if reflections < max_reflections:
            try:
                critique = generation_module.llm.invoke(
                    REFLEXION_PROMPT.format(question=question, answer=answer)
                ).content.strip()
                if critique.startswith("REVISE"):
                    context = generation_module._build_context(docs)
                    revised = generation_module.llm.invoke(
                        REVISE_PROMPT.format(
                            question=question, context=context, critique=critique
                        )
                    ).content
                    answer = revised
                    reflections += 1
                    logger.info(f"Reflexion: 回答已修订 ({critique[:80]})")
                else:
                    logger.info("Reflexion: 回答合格, 无需修订")
            except Exception as e:
                logger.warning(f"Reflexion 反思失败(保留原回答): {e}")

        return {"answer": answer, "reflections": reflections}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("generator", generator_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": "generator"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("generator", END)

    return graph.compile()
