"""Ollama client for MiniMax-M2.5:cloud model"""
import ollama
from typing import List, Optional


class OllamaClient:
    """Client for Ollama MiniMax-M2.5:cloud model"""

    def __init__(self, model: str = "minimax-m2.5:cloud", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)

    def check_model(self) -> bool:
        """Check if the model is available"""
        try:
            models = self.client.list()
            model_list = models.models if hasattr(models, 'models') else models
            for m in model_list:
                model_name = m.name if hasattr(m, 'name') else m.get('name', '')
                if self.model in model_name:
                    return True
            return False
        except Exception as e:
            print(f"Error checking model: {e}")
            return False

    def pull_model(self) -> bool:
        """Pull the MiniMax-M2.5:cloud model"""
        try:
            print(f"Pulling {self.model} model...")
            for progress in self.client.pull(self.model, stream=True):
                if 'status' in progress:
                    print(f"  {progress['status']}")
            return True
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False

    def chat(self, messages: List[dict], context: Optional[str] = None) -> str:
        """Generate a chat completion"""

        # If context is provided, prepend it to the system message
        if context:
            # Find system message or add one
            system_found = False
            for msg in messages:
                if msg.get('role') == 'system':
                    msg['content'] = f"{context}\n\n{msg['content']}"
                    system_found = True
                    break

            if not system_found:
                messages.insert(0, {
                    'role': 'system',
                    'content': f"You are a helpful AI assistant. Use the following context to answer questions:\n\n{context}"
                })

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_ctx': 8192,
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {e}"

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate a completion from a single prompt"""

        full_prompt = prompt
        if context:
            full_prompt = f"""Context information:
{context}

Question: {prompt}

Answer based on the context above:"""

        try:
            response = self.client.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_ctx': 8192,
                }
            )
            return response['response']
        except Exception as e:
            return f"Error: {e}"
