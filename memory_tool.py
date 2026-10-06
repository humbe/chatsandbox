from sys import exit
from langchain_mlx import MLXChatModel
from mem0 import Memory
from qdrant_client.http.exceptions import ResponseHandlingException

class MemoryWrapper:
    def __init__(self, mlx_chat):
        self.user_id = "default"
        try:
            self.memory = Memory.from_config({
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": "multi-qa-MiniLM-L6-cos-v1"
                    }
                },
                "llm": {
                    "provider": "langchain",
                    "config": {
                        "model": mlx_chat
                    }
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "demo",
                        "embedding_model_dims": 384,
                        "host": "localhost",
                        "port": 6333
                    }
                }
            })
        except ResponseHandlingException:
            print("Run the qdrant image locally: container run -d -p 6333:6333 qdrant/qdrant")
            exit()

    def store(self, mem:str):
        messages = [{
            "role": "user", "content": mem
        }]
        # don't infer, this breaks insert
        self.memory.add(messages, user_id=self.user_id, infer=False)

    def search(self, topic:str) -> str:
        results = self.memory.search(query=topic, filters={"user_id": self.user_id}, top_k=3)
        #print(f"memory.search results={results}")
        return "\n".join([m['memory'] for m in results['results']])

memory_instance = None

def init_memory_wrapper(tokenizer, model):
    global memory_instance
    mlx_chat = MLXChatModel(tokenizer=tokenizer, model=model)
    memory_instance = MemoryWrapper(mlx_chat)

def get_memory():
    return memory_instance
