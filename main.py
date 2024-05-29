from langchain_core.callbacks import CallbackManager
from langchain.callbacks.manager import Callbacks
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from src.applications import LlmApplications
from src.callback import StreamingStdOutCallbackHandler

import threading
import langchain
import chromadb
import argparse


from tools.retrieval import RetrievalQATool
from tools.weather import WeatherTool
from tools.chat import ChatTool

from langchain.globals import set_debug



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_path", type=str, default= 'models/ggml-model-Q4_K_M.gguf', help="local LLM path")
    parser.add_argument("--n_ctx", type=int, default=10000, help="number of tokens of context")
    parser.add_argument("--max_tokens", type=int, default=10000, help="max tokens of generated text")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature of LLM")
    parser.add_argument("--embedding_path", type=str, default='models/gte-small-zh', help="local embedding path")
    parser.add_argument("--reranker_path", type=str, default='models/bge-reranker-base', help="local reranker path")
    parser.add_argument("--db_path", type=str, default='faiss_db/luxun', help="vector database path")
    parser.add_argument("--docs_path", type=str, default='/home/pony/workspace/nlp/data/luxun', help="documents path")
    parser.add_argument("--stage1_top_k", type=int, default=20, help="top k chunks for stage 1 in retrieval use embedding model")
    parser.add_argument("--stage2_top_k", type=int, default=3, help="top k chunks for stage 2 in retrieval use reranker model")
    parser.add_argument("--debug", type=str, choices=['true', 'false'], default='true', help="debug mode")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if args.debug == 'true':
        set_debug(True)
    streamcallback = StreamingStdOutCallbackHandler()
    callback_manager = CallbackManager([streamcallback])
    consumer_thread = threading.Thread(target=streamcallback.consumer)
    consumer_thread.start()

    llm = LlamaCpp(
        model_path=args.llm_path,
        callback_manager=callback_manager,
        n_ctx=args.n_ctx,
        max_tokens = args.max_tokens,
        temperature = args.temperature,
        verbose=False
    )

    chat_tool = ChatTool(llm)
    retrieval_qa_tool = RetrievalQATool(args, llm)
    weather_tool = WeatherTool(llm)
    
    tools = [
            chat_tool, 
             weather_tool, 
             retrieval_qa_tool
             ]
    application = LlmApplications(tools, llm)
    application('帮我用维基百科查询一下太阳黑子运动的定义。')

if __name__ == "__main__":
    main()