from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from src.utils import Llama3PromptBuilder
from abc import ABC, abstractmethod


import sys


chat_prompt_template = """以下是人与人工智能之间的友好对话。人工智能健谈，并且从其对话历史中提供许多具体细节。如果人工智能不知道问题的答案，它会诚实地说它不知道。
当前对话：
{history}

问题：
{input}

回答："""

agent_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
When you find the keyword "final_answer" in the context, please return this as the Final Answer.

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

# agent_prompt_template = """用中文尽你所能回答问题，你可以使用下面的工具帮助你回答:

# {tools}

# 使用下面的格式:

# 问题: 输入的问题
# Thought: 你应该总是思考要做什么
# Action: 采取的决策, 应该是'[]'中的一种工具 [{tool_names}]
# Action Input: 工具的输入
# Observation: 从工具中得到的结果
# ... (这个 Thought/Action/Action Input/Observation 可以重复多次)
# Thought: 我知道最终答案了
# Final Answer: Question的最终答案

# 开始！

# 问题: {input}
# Thought: {agent_scratchpad}
# """

class BaseApplication(ABC):
    def __init__(self, llm, args):
        self.llm = llm
        self.args = args

    def __call__(self, input):
        try:
            res = self.chains.invoke({'input': input})
            output = res.get('output', "")
            if output == 'Agent stopped due to iteration limit or time limit.':
                sys.stdout.write('不好意思，响应超时，请重新输入。\n')
        except:
            sys.stdout.write('不好意思，响应超时，请重新输入。\n')

class ChatApplication(BaseApplication):
    def __init__(self, llm, args):
        super().__init__(llm, args)
        prompt = Llama3PromptBuilder('你是一个智能助手，你的名字叫小助。').build_chat_prompt(chat_prompt_template)
        self.chains = ConversationChain(
            llm=self.llm,
            memory=ConversationBufferMemory(),
            prompt=prompt
            )

class AgentApplication(BaseApplication):
    def __init__(self, tools, llm, args):
        super().__init__(llm, args)
        """
        tools: a list of Tool objects
        """
        self.tools = self._build_tools(tools)
        agent = create_react_agent(
                            llm=self.llm,
                            tools=self.tools,
                            prompt=PromptTemplate.from_template(agent_prompt_template),
                            stop_sequence=['\nObservation', '\n\nObservation', 'Observation']
                            )
        self.chains = AgentExecutor.from_agent_and_tools(agent=agent, tools=self.tools, max_iterations=self.args.agent_max_iters)
    
    def prepare():
        pass

    def _build_tools(self, tools):
        Tools = []
        for tool in tools:
            Tools.append(tool.tool_wrapper())
        return Tools
 
    
        
