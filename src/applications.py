from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate


agent_prompt = """Answer the following questions as best you can. You have access to the following tools:

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

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""


class LlmApplications():
    def __init__(self, tools, llm):
        """
        tools: a list of Tool objects
        """
        self.tools = self._build_tools(tools)
        agent = create_react_agent(
                            llm=llm,
                            tools=self.tools,
                            prompt=PromptTemplate.from_template(agent_prompt)
                            )
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools)

    def _build_tools(self, tools):
        Tools = []
        for tool in tools:
            Tools.append(tool.tool_wrapper())
        return Tools
 
    def __call__(self, input): 
        try:
            res = self.agent_executor.invoke({"input": input})
        except:
            res = {
                'output': '不好意思我没有听清你的问题，请重新输入。'
            }
        return res['output']
