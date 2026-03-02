"""MiniMax CLI Chatbot with RAG - Main entry point"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich import print as rprint
from rich.theme import Theme
import asyncio

from src.rag.vectorstore import VectorStore
from src.rag.web_search import WebSearch
from src.rag.document_loader import DocumentLoader
from src.llm.ollama_client import OllamaClient
from src.chat import ChatSession


# Custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
    "user": "bold blue",
    "bot": "bold cyan",
})

console = Console(theme=custom_theme)


class ChatBot:
    """CLI Chatbot with RAG using Ollama MiniMax-M2.5:cloud"""

    def __init__(self):
        self.llm = OllamaClient(model="minimax-m2.5:cloud")
        self.vectorstore = VectorStore()
        self.web_search = WebSearch()
        self.document_loader = DocumentLoader()
        self.session = ChatSession()

    def check_model(self) -> bool:
        """Check if model is available"""
        console.print("[info]Checking Ollama model...[/info]")

        if not self.llm.check_model():
            console.print("[warning]minimax-m2.5:cloud not found. Pulling now...[/warning]")
            if not self.llm.pull_model():
                console.print("[error]Failed to pull minimax-m2.5:cloud model[/error]")
                return False

        console.print("[success]Model ready![/success]")
        return True

    def search_web(self, query: str):
        """Search the web and add results to knowledge base"""
        console.print(f"[info]Searching web for: {query}[/info]")

        # Get search results and fetch content
        search_results = self.web_search.search(query, num_results=3)
        contents = self.web_search.search_and_fetch(query, num_results=3)

        if contents:
            # Add to vector store
            metadatas = [{'source': 'web_search', 'query': query} for _ in contents]
            self.vectorstore.add_documents(contents, metadatas)

            # Return both search metadata and contents for display
            return {'metadata': search_results, 'contents': contents}
        else:
            return None

    def load_documents(self, path: str) -> int:
        """Load documents into knowledge base"""
        path_obj = os.path.expanduser(path)

        if os.path.isfile(path_obj):
            chunks = self.document_loader.load_file(path_obj)
        elif os.path.isdir(path_obj):
            chunks = self.document_loader.load_directory(path_obj)
        else:
            console.print(f"[error]Path not found: {path}[/error]")
            return 0

        if chunks:
            contents = [c['content'] for c in chunks]
            metadatas = [c['metadata'] for c in chunks]
            self.vectorstore.add_documents(contents, metadatas)
            console.print(f"[success]Loaded {len(contents)} chunks from {path}[/success]")
            return len(contents)

        return 0

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context from knowledge base"""
        results = self.vectorstore.similarity_search(query, top_k)

        if results:
            context_parts = [r['content'] for r in results]
            return "\n\n".join(context_parts)
        return ""

    def chat(self, user_input: str) -> str:
        """Process a chat message with RAG"""
        # Add user message
        self.session.add_user_message(user_input)

        # Get messages
        messages = self.session.get_messages()

        # Build prompt with context from RAG
        context = self.retrieve_context(user_input)

        # Build the full prompt with context
        if context:
            # Update system message with context
            system_content = f"""You are a helpful AI assistant. Use the following context to answer the user's questions. If the context doesn't contain relevant information, you can use your general knowledge.

Context from knowledge base:
{context}

{self.session.system_prompt}"""
        else:
            system_content = self.session.system_prompt

        # Replace system message
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] = system_content
        else:
            messages.insert(0, {'role': 'system', 'content': system_content})

        # Call Ollama
        response = self.llm.chat(messages)

        # Add assistant response to session
        self.session.add_assistant_message(response)

        return response


def print_welcome():
    """Print welcome message"""
    welcome = """
[bold cyan]MiniMax ChatBot[/bold cyan] - CLI with RAG

[dim]Powered by MiniMax-M2.5:cloud + Web Search[/dim]

Commands:
  /search <query>    - Search the web and add to knowledge
  /load <path>      - Load documents into knowledge base
  /context          - Show current knowledge base info
  /clear            - Clear chat history
  /help             - Show this help message
  /exit             - Exit the chatbot

Type your message to start chatting!
"""
    console.print(Panel(welcome, border_style="cyan", title="MiniMax ChatBot"))


def run_interactive():
    """Run the interactive CLI"""
    bot = ChatBot()

    if not bot.check_model():
        return

    print_welcome()

    while True:
        try:
            user_input = console.input("\n[bold blue]You:[/bold blue] ")

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == '/exit':
                    console.print("[info]Goodbye![/info]")
                    break

                elif cmd == '/clear':
                    bot.session.clear()
                    console.print("[info]Chat history cleared[/info]")

                elif cmd == '/context':
                    count = bot.vectorstore.count()
                    console.print(f"[info]Knowledge base contains {count} documents[/info]")

                elif cmd == '/search':
                    if arg:
                        result = bot.search_web(arg)
                        if result:
                            contents = result['contents']
                            console.print(f"\n[bold green]Search results for '{arg}':[/bold green]\n")
                            for i, content in enumerate(contents, 1):
                                # Show the actual content, not just links
                                console.print(f"[bold]Result {i}:[/bold]")
                                # Limit display to a reasonable length
                                display_content = content[:1500] + "..." if len(content) > 1500 else content
                                console.print(f"{display_content}\n")
                                console.print("[dim]" + "="*50 + "[/dim]\n")
                        else:
                            console.print("[warning]No search results found.[/warning]")
                    else:
                        console.print("[warning]Usage: /search <query>[/warning]")

                elif cmd == '/load':
                    if arg:
                        bot.load_documents(arg)
                    else:
                        console.print("[warning]Usage: /load <path>[/warning]")

                elif cmd == '/help':
                    console.print("""
[bold]Commands:[/bold]
  /search <query>    - Search web and add to knowledge
  /load <path>      - Load documents (file or directory)
  /context          - Show knowledge base info
  /clear            - Clear chat history
  /exit             - Exit the chatbot
                    """)

                else:
                    console.print(f"[warning]Unknown command: {cmd}[/warning]")

                continue

            # Regular chat
            with console.status("[bold cyan]Thinking...", spinner="dots"):
                response = bot.chat(user_input)

            # Print response
            console.print("\n[bold cyan]Bot:[/bold cyan]")
            # Try to render as markdown
            try:
                md = Markdown(response)
                console.print(md)
            except:
                console.print(response)

        except KeyboardInterrupt:
            console.print("\n[info]Use /exit to quit[/info]")
        except Exception as e:
            console.print(f"[error]Error: {e}[/error]")


if __name__ == '__main__':
    run_interactive()
