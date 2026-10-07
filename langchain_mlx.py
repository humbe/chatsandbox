#import mlx.core as mx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from mlx_lm import generate
from typing import Any, List, Optional, Dict
import json

class MLXChatModel(BaseChatModel):

    model: Any = None
    tokenizer: Any = None

    def __init__(self, tokenizer: Any, model: Any):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs) -> ChatResult:
        mlx_messages = []

        for msg in messages:
            if msg.type == "human":
                mlx_messages.append({
                    "role": "user",
                    "content": msg.content
                })
            elif msg.type == "ai":
                mlx_messages.append({
                    "role": "assistant",
                    "content": msg.content
                })
            elif msg.type == "system":
                mlx_messages.append({
                    "role": "system",
                    "content": msg.content
                })

        print(f"mem0 asked {mlx_messages}")

        prompt = self.tokenizer.apply_chat_template(mlx_messages, tokenize=False, add_generation_prompt=True)

        response_text = generate(model=self.model, tokenizer=self.tokenizer, prompt=prompt, max_tokens=1024)

        print(f"mlx responded {response_text}")

        # Wrap in LangChain format
        message = AIMessage(content=response_text.strip())
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mlx-chat-model"
