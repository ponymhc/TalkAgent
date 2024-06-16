from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from src.utils import Llama3PromptBuilder

conversation_prompt_template = """
以下是你与人类先前的友好对话。你非常健谈且会提供大量具体细节。如果你不知道问题的答案，它会如实地说不知道。

历史对话：
{history}

新输入:
{input}
"""


class Conversation:
    def __init__(self, llm, application):
        self.prompt_builder = Llama3PromptBuilder('你是一个智能聊天助手。')
        self.conversation_prompt = self.prompt_builder.build_chat_prompt(conversation_prompt_template)
        self.memory = ConversationSummaryBufferMemory(llm=llm)
        self.chain = ConversationChain(llm=llm, memory=ConversationSummaryBufferMemory(llm=llm), prompt=self.conversation_prompt)
        self.application = application
    
    def __call__(self, input):
        self.application(input)