from src.utils import Llama3PromptBuilder
from langchain.agents import Tool

class ChatTool():
    def __init__(self, llm):
        self.llm = llm
        self.prompt_builder = Llama3PromptBuilder('你是一个聊天机器人，可以用中文回答各种问题。')
        self.prompt = self.prompt_builder.build_common_prompt()
        self.chain = self.prompt | self.llm

    def tool_wrapper(self):
        tool = Tool(
            name='Language Model',
            func=self.chain.invoke,
            description='用于聊天。'
        )
        return tool
    