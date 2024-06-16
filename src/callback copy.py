from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List, Optional
from espnet2.bin.tts_inference import Text2Speech
import cn2an
import numpy as np
import threading
import pyaudio
import queue
import sys
import re


DEFAULT_ANSWER_PREFIX_TOKENS = ["Final", "Answer", ":"]

class BaseStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__():
        pass

class ChatStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__():
        pass

class AgentFinalStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    """Callback handler for streaming in agents.
    Only works with agents using LLMs that support streaming.

    Only the final output of the agent will be streamed.
    """
    
    def append_to_last_tokens(self, token: str) -> None:
        self.last_tokens.append(token)
        self.last_tokens_stripped.append(token.strip())
        if len(self.last_tokens) > len(self.answer_prefix_tokens):
            self.last_tokens.pop(0)
            self.last_tokens_stripped.pop(0)

    def check_if_answer_reached(self) -> bool:
        if self.strip_tokens:
            return self.last_tokens_stripped == self.answer_prefix_tokens_stripped
        else:
            return self.last_tokens == self.answer_prefix_tokens

    def __init__(
        self,
        *,
        answer_prefix_tokens: Optional[List[str]] = None,
        strip_tokens: bool = True,
        stream_prefix: bool = False,
        args = None
    ) -> None:
        """Instantiate FinalStreamingStdOutCallbackHandler.

        Args:
            answer_prefix_tokens: Token sequence that prefixes the answer.
                Default is ["Final", "Answer", ":"]
            strip_tokens: Ignore white spaces and new lines when comparing
                answer_prefix_tokens to last tokens? (to determine if answer has been
                reached)
            stream_prefix: Should answer prefix itself also be streamed?
        """
        super().__init__()
        if answer_prefix_tokens is None:
            self.answer_prefix_tokens = DEFAULT_ANSWER_PREFIX_TOKENS
        else:
            self.answer_prefix_tokens = answer_prefix_tokens
        if strip_tokens:
            self.answer_prefix_tokens_stripped = [
                token.strip() for token in self.answer_prefix_tokens
            ]
        else:
            self.answer_prefix_tokens_stripped = self.answer_prefix_tokens

        self.last_tokens = [""] * len(self.answer_prefix_tokens)
        self.last_tokens_stripped = [""] * len(self.answer_prefix_tokens)
        self.strip_tokens = strip_tokens
        self.stream_prefix = stream_prefix
        self.answer_reached = False

        self.args=args
        self.t2s = Text2Speech(model_file=f"{args.tts_model_path}/train.total_count.ave_10best.pth",
                    train_config=f"{args.tts_model_path}/config.yaml",
                    noise_scale=0.667,
                    noise_scale_dur=0.8,
                    speed_control_alpha=0.9)
        self.token_cache = ""
        self.message_queue = queue.Queue(maxsize=1000)
        self.pattern = re.compile(r'[^\d\u4e00-\u9fa5，。！？,.!?]')
        self.RATE=self.t2s.fs
        self.CHANNELS=1
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        output=True)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.answer_reached = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

        # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.append_to_last_tokens(token)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.check_if_answer_reached():
            self.answer_reached = True
            if self.stream_prefix:
                for t in self.last_tokens:
                    sys.stdout.write(t)
                sys.stdout.flush()
            return

        # ... if yes, then print tokens from now on
        if self.answer_reached:
            sys.stdout.write(token)
            sys.stdout.flush()

            # 分流到音频输出
            matched = re.finditer('[，：。？！!?]', token)
            punctuation_indices = [m.start() for m in matched]
            self.token_cache += token
            if punctuation_indices:
                absolute_index = punctuation_indices[0] + len(self.token_cache) - len(token)
                self.message_queue.put(self.token_cache[:absolute_index + 1])
                self.token_cache = self.token_cache[absolute_index + 1:]
    
    def _number2char(self, text):
        pattern = r'-?\d+(\.\d+)?'

        def increment_number(match):
            num_str = match.group(0)
            return cn2an.an2cn(num_str)

        new_text = re.sub(pattern, increment_number, text)
        return new_text

    def clean_text(self, text):
        text = text.replace('\n','').replace('，','。').replace('：','。')
        text = self.pattern.sub('', text)
        text = self._number2char(text)
        return text
    
    def consumer(self):
        while True:
            message = self.message_queue.get()
            clean_message = self.clean_text(message)
            wav = self.t2s(clean_message)['wav'].view(-1).cpu().numpy()
            audio_data_pcm = (wav * 32767).astype(np.int16)
            self.stream.write(audio_data_pcm.tobytes())