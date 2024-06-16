from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS, faiss
from langchain.chains import RetrievalQA
from langchain.agents import Tool

from src.utils import Llama3PromptBuilder
from tools.reranker import Reranker

import os

retrieval_prompt_template = """
请根据下面的上下文内容回答问题：

上下文：
{context}

问题：
{question}

注意：回答问题请根据上下文内容，如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，
不允许在答案中添加编造成分，答案请使用中文。
"""

class RetrievalQATool:
    def __init__(self, llm, args):
        self.conf = args
        self.prompt_builder = Llama3PromptBuilder('你是一个智能机器人，可以检索文档并根据检索结果回答问题。')
        self.prompt = self.prompt_builder.build_chat_prompt(retrieval_prompt_template)
        self.embedding = SentenceTransformerEmbeddings(model_name=self.conf.embedding_path)
        self.db = self._init_db()
        self.retriever = self.db.as_retriever(search_kwargs={'top_k': self.conf.stage1_top_k})
        self.reranker = Reranker(self.conf.reranker_path, args.stage2_top_k)
        self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker, base_retriever=self.retriever
            )
        self.chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=self.compression_retriever,
            chain_type_kwargs={
                'prompt': self.prompt
            },
        )
    
    def _init_db(self):
        if not os.path.isdir(self.conf.db_path):
            print(self.conf.db_path)
            try:
                docs = []
                for doc in os.listdir(self.conf.docs_path):
                    if doc.endswith('.txt'):
                        loader = TextLoader(f'{self.conf.docs_path}/{doc}')
                        doc = loader.load()
                        docs.extend(doc)
                text_splitter=CharacterTextSplitter(
                                        chunk_size=128, 
                                        chunk_overlap=8, 
                                        separator='\n'
                                    )
                docs = text_splitter.split_documents(docs)
                db = FAISS.from_documents(docs, 
                                          self.embedding
                                          )
                db.save_local(self.conf.db_path)
            except:
                print('初始化失败...')
                if os.path.isdir(self.conf.db_path):
                    os.rmdir(self.conf.db_path)
        else:
            db = FAISS.load_local(self.conf.db_path, self.embedding, allow_dangerous_deserialization=True)
        return db

    def tool_wrapper(self):
        tool = Tool(
            name='Retriever',
            func=self.chain.invoke,
            description='用于知识库检索问答。'
        )
        return tool
