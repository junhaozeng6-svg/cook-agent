"""
RAG系统主程序
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self._agent_graph = None   # Agent模式: LangGraph状态图(懒加载)
        self._history = []         # Agent模式: 多轮对话历史

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

    #初始化所有模块，将config文件
    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载食谱文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")
    
    def _fast_path_answer(self, question: str, stream: bool = False):
        """
        标准管线回答(原ask_question逻辑) - Fast Path

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        
        print(f"\n❓ 用户问题: {question}")

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'list':
            # 列表查询保持原查询
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        else:
            # 详细查询和一般查询使用智能重写
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)
        
        # 3. 检索相关子块（自动应用元数据过滤）
        print("🔍 检索相关文档...")
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k=self.config.top_k)
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)

        # 显示检索到的子块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get('dish_name', '未知菜品')
                # 尝试从内容中提取章节标题
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith('#'):
                    # 如果是标题开头，提取标题（仅取第一行）
                    title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                    section_title = content_preview[:title_end].replace('#', '').strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        # 5. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：直接返回菜品名称列表
            print("📋 生成菜品列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")

            return self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            # 详细查询：获取完整文档并生成详细回答
            print("获取完整文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")
            else:
                print(f"对应 {len(relevant_docs)} 个完整文档")

            print("✍️ 生成详细回答...")

            # 根据路由类型自动选择回答模式
            if route_type == "detail":
                # 详细查询使用分步指导模式
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_step_by_step_answer(question, relevant_docs)
            else:
                # 一般查询使用基础回答模式
                if stream:
                    return self.generation_module.generate_basic_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_basic_answer(question, relevant_docs)

    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题 - Fast Path / ReAct 双路径

        Args:
            question: 用户问题
            stream: 是否使用流式输出(流式暂走标准管线)

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        if stream:
            # 流式输出: 暂走确定性标准管线(ReAct路径暂不支持流式)
            return self._fast_path_answer(question, stream=True)

        # 记录本轮问题到多轮历史(供Agent模式跨轮指代)
        self._history.append(HumanMessage(content=question))
        self._trim_history()

        # Router节点: 轻量LLM判断走哪条路径
        route = self._route_question(question)
        if route == "react":
            print("🤖 进入Agent模式(ReAct循环)...")
            answer = self._react_answer(question)
        else:
            answer = self._fast_path_answer(question, stream=False)

        # 记录回答到历史
        if isinstance(answer, str):
            self._history.append(AIMessage(content=answer))
        return answer

    def _route_question(self, question: str) -> str:
        """
        Router节点: 轻量LLM判断问题复杂度, 决定走 Fast Path 还是 ReAct 循环

        Returns:
            'fast' 或 'react'
        """
        prompt = ChatPromptTemplate.from_template("""
你是查询路由。判断该烹饪问题应该走哪条处理路径, 只输出JSON:

- "fast": 问题明确具体(含具体菜名/食材/做法关键词), 标准管线一次处理即可
- "react": 问题模糊、多条件组合(多个筛选)、口语化、或需要多步推理/可能检索失败需要重试

输出格式: {{"route": "fast"}} 或 {{"route": "react"}}

用户问题: {question}""")

        chain = prompt | self.generation_module.llm | StrOutputParser()
        try:
            raw = chain.invoke(question).strip()
            route = json.loads(raw).get("route", "fast")
        except Exception:
            route = "fast"  # 解析失败时保守走标准管线
        if route not in ("fast", "react"):
            route = "fast"
        logger.info(f"Router决策: {route}")
        return route

    def _react_answer(self, question: str) -> str:
        """
        Slow Path: 多智能体编排 - 主管-工人模式
        (Supervisor 委派给 菜谱Agent / 推荐Agent, 各自独立 ReAct 循环)

        Args:
            question: 用户问题

        Returns:
            最终回答
        """
        if self._agent_graph is None:
            from rag_modules.agent_supervisor import build_supervisor_graph
            self._agent_graph = build_supervisor_graph(
                retrieval_module=self.retrieval_module,
                data_module=self.data_module,
                generation_module=self.generation_module,
                top_k=self.config.top_k,
                max_steps=3,
                max_reflections=1,
                max_delegations=2,
            )

        system_prompt = SystemMessage(content=(
            "你是'尝尝咸淡'烹饪助手的主管Agent, 手下有两位专家: "
            "菜谱Agent(擅长具体做法/食材/步骤, 委派 delegate_to_recipe_worker)和 "
            "推荐Agent(擅长推荐菜品/按分类难度筛选, 委派 delegate_to_recommender_worker)。"
            "根据用户问题委派给合适的专家; 专家返回结果后, 基于结果组织最终回答。"
            "若问题涉及多个方面, 可以多次委派不同专家。信息足够后直接给出最终回答。"
        ))

        result = self._agent_graph.invoke({
            "messages": [system_prompt] + self._history,
            "delegations": 0,
        })
        return result.get("answer", "抱歉，未能生成回答。")

    def _trim_history(self, max_rounds: int = 6):
        """
        多轮记忆窗口管理: 保留最近 max_rounds 轮问答, 超长单条消息截断,
        防止上下文无限增长导致 token 超限。

        Args:
            max_rounds: 保留的对话轮数(每轮 = 1个问题 + 1个回答)
        """
        max_msgs = max_rounds * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

        # 超长单条消息截断
        for msg in self._history:
            content = getattr(msg, "content", "")
            if isinstance(content, str) and len(content) > 2000:
                msg.content = content[:2000] + "...(已截断)"

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters
    
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                
                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用尝尝咸淡RAG系统！")


def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()
