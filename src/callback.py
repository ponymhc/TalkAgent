from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List, Optional
import queue
import sys
import re

DEFAULT_ANSWER_PREFIX_TOKENS = ["Final", "Answer", ":"]

# class StreamingStdOutCallbackHandler(BaseCallbackHandler):
#     """Callback handler for streaming. Only works with LLMs that support streaming."""
#     token_cache = ""
#     message_queue = queue.Queue(maxsize=1000)
#     def on_llm_start(
#         self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
#     ) -> None:
#         """Run when LLM starts running."""

#     def on_chat_model_start(
#         self,
#         serialized: Dict[str, Any],
#         messages: List[List[BaseMessage]],
#         **kwargs: Any,
#     ) -> None:
#         """Run when LLM starts running."""

#     def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
#         """Run on new LLM token. Only available when streaming is enabled."""
#         sys.stdout.write(token)
#         sys.stdout.flush()
#         matched = re.finditer('[,.，。？！!?]', token)
#         punctuation_indices = [m.start() for m in matched]
#         self.token_cache += token
#         if punctuation_indices:
#             absolute_index = punctuation_indices[0] + len(self.token_cache) - len(token)
#             self.message_queue.put(self.token_cache[:absolute_index + 1])
#             self.token_cache = self.token_cache[absolute_index + 1:]

#     def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
#         """Run when LLM ends running."""

#     def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
#         """Run when LLM errors."""

#     def on_chain_start(
#         self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
#     ) -> None:
#         """Run when chain starts running."""

#     def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
#         """Run when chain ends running."""

#     def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
#         """Run when chain errors."""

#     def on_tool_start(
#         self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
#     ) -> None:
#         """Run when tool starts running."""

#     def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
#         """Run on agent action."""
#         pass

#     def on_tool_end(self, output: Any, **kwargs: Any) -> None:
#         """Run when tool ends running."""

#     def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
#         """Run when tool errors."""

#     def on_text(self, text: str, **kwargs: Any) -> None:
#         """Run on arbitrary text."""

#     def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
#         """Run on agent end."""

#     def consumer(self):
#         while True:
#             message = self.message_queue.get().replace('\n', '')
#             print(f"Consumed: {message}")

class FinalStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    """Callback handler for streaming in agents.
    Only works with agents using LLMs that support streaming.

    Only the final output of the agent will be streamed.
    """
    token_cache = ""
    message_queue = queue.Queue(maxsize=1000)
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
            matched = re.finditer('[,.，。？！!?]', token)
            punctuation_indices = [m.start() for m in matched]
            self.token_cache += token
            if punctuation_indices:
                absolute_index = punctuation_indices[0] + len(self.token_cache) - len(token)
                self.message_queue.put(self.token_cache[:absolute_index + 1])
                self.token_cache = self.token_cache[absolute_index + 1:]
    
    def consumer(self):
        while True:
            message = self.message_queue.get().replace('\n', '')
            print(f"Consumed: {message}")
