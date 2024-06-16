from src.callback import AgentFinalStreamingStdOutCallbackHandler, ChatStreamingStdOutCallbackHandler
from langchain_core.callbacks import CallbackManager
from langchain_community.llms import LlamaCpp
from src.applications import AgentApplication, ChatApplication
from src.asr import AudioRecorder, Transcriber

import logging
import threading
import langchain
import argparse
import sys
import time


from tools.retrieval import RetrievalQATool
from tools.weather import WeatherTool
from tools.message_sender import EmailTool

from langchain.globals import set_debug, set_verbose

TOOLS = {
    WeatherTool,
    RetrievalQATool,
    EmailTool
}

CALLBACKS = {
    'chat': ChatStreamingStdOutCallbackHandler,
    'agent' : AgentFinalStreamingStdOutCallbackHandler
}

def conversation(application, args):
    try:
        with AudioRecorder(channels=1, sample_rate=16000) as recorder:
            with Transcriber(model_size=args.asr_model_path) as transcriber:
                for audio in recorder:
                    for seg in transcriber(audio):
                        print('> ' + seg)
                        try:
                            application(seg)
                        except:
                            time.sleep(1)
                            print("**************************请不要密集提问！**************************")
                        print()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: terminating...")
    except Exception as e:
        logging.error(e, exc_info=True, stack_info=True)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=['chat', 'agent'], default= 'agent', help="chat or agent mode")
    parser.add_argument("--llm_path", type=str, default= 'models/ggml-model-q8_0.gguf', help="local LLM path")
    parser.add_argument("--n_gpu_layers", type=int, default=-1, help="number of layers of llm on gpu inference")
    parser.add_argument("--n_ctx", type=int, default=10000, help="number of tokens of context")
    parser.add_argument("--max_tokens", type=int, default=10000, help="max tokens of generated text")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature of LLM")
    parser.add_argument("--asr_model_path", type=str, default= 'models/faster-whisper-small', help="asr model dir")
    parser.add_argument("--tts_model_path", type=str, default= 'models/vits', help="tts model dir")
    parser.add_argument("--embedding_path", type=str, default='models/gte-large-zh', help="local embedding path")
    parser.add_argument("--reranker_path", type=str, default='models/bge-reranker-base', help="local reranker path")
    parser.add_argument("--db_path", type=str, default='faiss_db/luxun', help="vector database path")
    parser.add_argument("--docs_path", type=str, default='data/luxun', help="documents path")
    parser.add_argument("--stage1_top_k", type=int, default=20, help="top k chunks for stage 1 in retrieval use embedding model")
    parser.add_argument("--stage2_top_k", type=int, default=3, help="top k chunks for stage 2 in retrieval use reranker model")
    parser.add_argument("--agent_max_iters", type=int, default=5, help="the maximum number of self ask iterations for agent")
    parser.add_argument("--debug", type=str, choices=['true', 'false'], default='false', help="debug mode")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if args.debug == 'true':
        set_debug(True)
        set_verbose(True)
    else:
        set_debug(False)
        set_verbose(False)

    streamcallback = CALLBACKS[args.mode](args=args)
    callback_manager = CallbackManager([streamcallback])
    consumer_thread = threading.Thread(target=streamcallback.consumer)
    consumer_thread.start()

    llm = LlamaCpp(
        model_path=args.llm_path,
        callback_manager=callback_manager,
        n_ctx=args.n_ctx,
        max_tokens = args.max_tokens,
        temperature = args.temperature,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False
    )

    if args.mode == 'chat':
        application = ChatApplication(llm, args)
    elif args.mode == 'agent':
        tools = [tool(llm, args) for tool in TOOLS]
        application = AgentApplication(tools, llm, args)
    conversation(application, args)

if __name__ == "__main__":
    main()