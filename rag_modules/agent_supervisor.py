"""
多智能体编排层 - 主管-工人(Supervisor-Workers)模式

图结构:
    START → supervisor(主管LLM, 绑定委派工具)
      ├─ delegate_to_recipe_worker      → recipe_worker(菜谱Agent: 独立ReAct循环)
      ├─ delegate_to_recommender_worker → recommender_worker(推荐Agent: 独立ReAct循环)
      └─ 无委派 / 委派次数超限          → finalize(汇总回答) → END
    worker 完成后结果回填 supervisor → 继续决策(可再次委派或直接回答)

设计要点:
    - 每个 Worker 是独立的 ReAct 子图(agent ⇄ tools → generator, 含Reflexion),
      只绑定自己的工具子集, 形成"角色隔离"
    - Supervisor 通过工具调用(委派)做路由, 观察 Worker 结果后汇总
    - delegations ≥ max_delegations 强制收尾, 防无限委派循环
"""

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .agent_graph import build_agent_graph

logger = logging.getLogger(__name__)

DEFAULT_MAX_DELEGATIONS = 2

# 每个 Worker 的工具子集(角色隔离)
RECIPE_WORKER_TOOLS = ["search_recipe", "get_full_recipe", "rewrite_query"]
RECOMMENDER_WORKER_TOOLS = ["filter_by_category", "list_categories", "search_recipe"]


@tool
def delegate_to_recipe_worker(question: str) -> str:
    """将问题转交给菜谱专家Agent处理。适用于询问具体做法、所需食材、制作步骤、烹饪技巧等需要完整菜谱内容的问题。question为需要转交的原始用户问题。"""
    return f"已转交菜谱Agent: {question}"


@tool
def delegate_to_recommender_worker(question: str) -> str:
    """将问题转交给推荐专家Agent处理。适用于推荐菜品、有哪些菜、按分类或难度筛选菜名等需要菜名列表的问题。question为需要转交的原始用户问题。"""
    return f"已转交推荐Agent: {question}"


class SupervisorState(TypedDict):
    """主管图状态: 消息(含委派记录) + 委派计数 + 最终答案"""
    messages: Annotated[list, add_messages]
    delegations: int
    answer: str


def build_supervisor_graph(retrieval_module, data_module, generation_module,
                           top_k: int = 3, max_steps: int = 3,
                           max_reflections: int = 1,
                           max_delegations: int = DEFAULT_MAX_DELEGATIONS):
    """
    构建主管-工人多智能体状态图。

    Args:
        retrieval_module:   RetrievalOptimizationModule 实例
        data_module:        DataPreparationModule 实例
        generation_module:  GenerationIntegrationModule 实例(提供 llm 与生成函数)
        top_k:              检索候选数量
        max_steps:          Worker 内部 ReAct 循环最大步数
        max_reflections:    Worker 生成后反思最大次数
        max_delegations:    主管最多委派次数(防无限委派)

    Returns:
        编译好的 CompiledStateGraph
    """
    # 两个 Worker: 独立 ReAct 子图, 只绑定各自的工具
    recipe_worker = build_agent_graph(
        retrieval_module, data_module, generation_module,
        top_k=top_k, max_steps=max_steps, max_reflections=max_reflections,
        tool_names=RECIPE_WORKER_TOOLS,
    )
    recommender_worker = build_agent_graph(
        retrieval_module, data_module, generation_module,
        top_k=top_k, max_steps=max_steps, max_reflections=max_reflections,
        tool_names=RECOMMENDER_WORKER_TOOLS,
    )

    supervisor_llm = generation_module.llm.bind_tools(
        [delegate_to_recipe_worker, delegate_to_recommender_worker]
    )

    def supervisor_node(state: SupervisorState) -> dict:
        """主管节点: LLM 决定委派给哪个 Worker, 或直接给出最终回答"""
        response = supervisor_llm.invoke(state["messages"])
        had_delegation = 1 if (hasattr(response, "tool_calls") and response.tool_calls) else 0
        return {
            "messages": [response],
            "delegations": state.get("delegations", 0) + had_delegation,
        }

    def make_worker_node(worker_graph, worker_name: str):
        """Worker节点工厂: 运行独立ReAct子图, 结果作为ToolMessage回填给主管"""
        def worker_node(state: SupervisorState) -> dict:
            last = state["messages"][-1]
            tc = last.tool_calls[0]
            question = tc.get("args", {}).get("question", "")
            result = worker_graph.invoke({
                "messages": [HumanMessage(content=question)],
                "step": 0,
                "reflections": 0,
            })
            worker_answer = result.get("answer", "抱歉，未能生成回答。")
            return {
                "messages": [ToolMessage(
                    content=worker_answer,
                    tool_call_id=tc.get("id", ""),
                    name=tc.get("name", worker_name),
                )]
            }
        return worker_node

    def route_from_supervisor(state: SupervisorState) -> str:
        """委派路由: 读主管的工具调用名分发到对应Worker; 无委派或超限则收尾"""
        if state.get("delegations", 0) >= max_delegations:
            logger.info(f"主管委派次数达到上限 {max_delegations}, 收尾")
            return "end"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            name = last.tool_calls[0].get("name", "")
            if name == "delegate_to_recipe_worker":
                return "recipe_worker"
            if name == "delegate_to_recommender_worker":
                return "recommender_worker"
        return "end"

    def finalize_node(state: SupervisorState) -> dict:
        """收尾节点: 提取主管的最终回答文本写入 answer"""
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                return {"answer": m.content}
        return {"answer": "抱歉，未能生成回答。"}

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("recipe_worker", make_worker_node(recipe_worker, "recipe_worker"))
    graph.add_node("recommender_worker", make_worker_node(recommender_worker, "recommender_worker"))
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "recipe_worker": "recipe_worker",
            "recommender_worker": "recommender_worker",
            "end": "finalize",
        },
    )
    graph.add_edge("recipe_worker", "supervisor")
    graph.add_edge("recommender_worker", "supervisor")
    graph.add_edge("finalize", END)

    return graph.compile()
