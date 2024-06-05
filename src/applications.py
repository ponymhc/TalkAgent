from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

import sys


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

Notice: When the Observation corresponds to the answer to the Question, please refine the Final Answer based on the Observation.

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""


class LlmApplications():
    def __init__(self, tools, llm, args):
        """
        tools: a list of Tool objects
        """
        self.conf = args
        self.tools = self._build_tools(tools)
        agent = create_react_agent(
                            llm=llm,
                            tools=self.tools,
                            prompt=PromptTemplate.from_template(agent_prompt_template),
                            stop_sequence=['\nObservation', '\n\nObservation', 'Observation']
                            )
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=self.tools, memory=self.memory, max_iterations=self.conf.agent_max_iters)

    def _build_tools(self, tools):
        Tools = []
        for tool in tools:
            Tools.append(tool.tool_wrapper())
        return Tools
 
    def __call__(self, input): 
        try:
            res = self.agent_executor.invoke({"input": input})
            if res['output'] == 'Agent stopped due to iteration limit or time limit.':
                sys.stdout.write('不好意思，响应超时，请重新输入。\n')
        except:
            sys.stdout.write('不好意思，响应超时，请重新输入。\n')
        
