import os
import logging
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import json
from langgraph.types import Command, interrupt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# === CONFIGURATION ===
MODEL = os.getenv("CHATBOT_MODEL", "gemini-pro")
API_KEY = os.getenv("CHATBOT_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Validate required environment variables
IS_DEV = os.getenv("ENV", "production").lower() == "dev"
if not API_KEY:
    if IS_DEV:
        logger.warning("CHATBOT_API_KEY environment variable is missing (dev mode)")
    else:
        raise ValueError("CHATBOT_API_KEY environment variable is required")
if not TAVILY_API_KEY:
    if IS_DEV:
        logger.warning("TAVILY_API_KEY environment variable is missing (dev mode)")
    else:
        raise ValueError("TAVILY_API_KEY environment variable is required")

# Set the correct environment variable for the selected model
if MODEL.startswith("openai:"):
    os.environ["OPENAI_API_KEY"] = API_KEY
    model_provider = "openai"
elif MODEL.startswith("anthropic:"):
    os.environ["ANTHROPIC_API_KEY"] = API_KEY
    model_provider = "anthropic"
elif MODEL.startswith("google:") or MODEL.startswith("gemini"):
    os.environ["GOOGLE_API_KEY"] = API_KEY
    model_provider = "google_genai"
else:
    model_provider = "google_genai"  # Default fallback
    logger.warning(f"Unknown model prefix for {MODEL}, using default provider")

# Set Tavily API key
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

logger.info(f"Using model: {MODEL}")
# === END CONFIGURATION ===

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human when the AI needs help with complex or sensitive queries."""
    # This will raise a Command exception that gets caught by the stream handler
    return interrupt({"query": query})

@tool
def generate_linkedin_post(topic: str, style: str = "professional") -> str:
    """Generate a LinkedIn post based on topic and style preferences"""
    prompt = f"""
    Create a LinkedIn post about {topic} in a {style} style.
    Structure: Hook → Story/Insight → Key takeaway → Call to action
    Keep it under 300 words, use 2-3 relevant hashtags.
    """
    llm = init_chat_model(MODEL, model_provider=model_provider)
    response = llm.invoke(prompt)
    return response.content

@tool
def generate_twitter_thread(topic: str, num_tweets: int = 5) -> str:
    """Generate a Twitter thread on the given topic"""
    prompt = f"""
    Create a {num_tweets}-tweet thread about {topic}.
    Start with a hook, provide value in middle tweets, end with engagement.
    Each tweet max 280 chars. Number them 1/{num_tweets}, 2/{num_tweets}, etc.
    """
    llm = init_chat_model(MODEL, model_provider=model_provider)
    response = llm.invoke(prompt)
    return response.content

@tool
def post_to_linkedin(content: str) -> str:
    """Stub: Post content to LinkedIn via API (not implemented)."""
    try:
        LINKEDIN_CLIENT_ID = os.getenv('LINKEDIN_CLIENT_ID')
        LINKEDIN_CLIENT_SECRET = os.getenv('LINKEDIN_CLIENT_SECRET')
        if not LINKEDIN_CLIENT_ID or not LINKEDIN_CLIENT_SECRET:
            return "LinkedIn API credentials not configured. Please set LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET in your .env file."
        # NOTE: Actual LinkedIn API posting is not implemented in this MVP.
        return f"Successfully posted to LinkedIn:\n{content}"
    except Exception as e:
        return f"Error posting to LinkedIn: {str(e)}"

@tool
def schedule_content(content: str, platform: str, datetime: str) -> str:
    """Schedule content for later posting"""
    return f"Content scheduled for {platform} at {datetime}:\n{content}"

@tool
def browse_web_page(url: str) -> str:
    """Browses a web page and returns its text content.

    Args:
        url: The URL of the web page to browse.

    Returns:
        The text content of the web page, or an error message if fetching fails.
    """
    if not url or not url.startswith(('http://', 'https://')):
        return "Invalid URL. Please provide a full and valid URL starting with http:// or https://."

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
            
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:5000] # Return the first 5000 characters to avoid being too long
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

class ConversationalAgent:
    """Main chatbot class with improved error handling and state management"""

    def __init__(self):
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        logger.info("Conversational agent initialized successfully")

    def _build_graph(self):
        """Build the conversation graph with tools and a system prompt"""
        graph_builder = StateGraph(State)

        # Set up tools
        tavily_search = TavilySearch(max_results=2)
        tools = [
            tavily_search, 
            human_assistance, 
            browse_web_page,
            generate_linkedin_post,
            generate_twitter_thread,
            post_to_linkedin,
            schedule_content
        ]

        # Initialize LLM with tools
        try:
            llm = init_chat_model(MODEL, model_provider=model_provider)
            llm_with_tools = llm.bind_tools(tools, enforce_single_tool=True)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

        # Create a system prompt to guide the LLM's tool usage
        system_prompt = (
            "You are a helpful assistant specializing in content creation and research. You have access to:\n"
            "1. 'tavily_search' - find information on the web\n"
            "2. 'browse_web_page' - read specific URLs\n"
            "3. 'generate_linkedin_post' - create LinkedIn posts\n"
            "4. 'generate_twitter_thread' - create Twitter threads\n"
            "5. 'post_to_linkedin' - publish to LinkedIn\n"
            "6. 'schedule_content' - schedule posts\n"
            "7. 'human_assistance' - request human help\n\n"
            "For content creation: Research topic first, then generate appropriate format.\n"
            "Always ask for clarification on tone, audience, and posting preferences.\n\n"
            "- If the user provides a URL, use the 'browse_web_page' tool to read its content.\n"
            "- If the user asks you to browse a page without providing a URL, you MUST ask for one.\n"
            "- For general questions, use the 'tavily_search' tool.\n"
            "- If the user asks for 'expert guidance', 'human help', or explicitly asks you to 'request assistance', "
            "you MUST use the 'human_assistance' tool. Do not try to answer these queries yourself."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                ("placeholder", "{messages}"),
            ]
        )

        # Create a chain that combines the prompt and the LLM
        agent_chain = prompt | llm_with_tools

        # Define chatbot node
        def chatbot_node(state: State):
            try:
                filtered_messages = self._filter_messages(state["messages"])
                if not filtered_messages:
                    logger.warning("No valid messages found in state")
                    return {"messages": []}
                message = agent_chain.invoke({"messages": filtered_messages})
                if hasattr(message, "tool_calls") and len(message.tool_calls) > 1:
                    logger.warning(f"Multiple tool calls detected: {len(message.tool_calls)}")
                return {"messages": [message]}
            except Exception as e:
                logger.error(f"Error in chatbot node: {e}")
                error_msg = {"role": "assistant", "content": "I encountered an error processing your request. Please try again."}
                return {"messages": [error_msg]}

        # Build graph
        graph_builder.add_node("chatbot", chatbot_node)
        
        tool_node = ToolNode(tools=tools)
        graph_builder.add_node("tools", tool_node)
        
        # Add edges
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")
        
        return graph_builder.compile(checkpointer=self.memory)

    def _filter_messages(self, messages):
        """Filter messages to keep only valid ones with improved logic"""
        filtered_messages = []
        
        for msg in messages:
            # Handle different message types
            if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls", None):
                # Keep tool call messages
                filtered_messages.append(msg)
            # CORRECTED: Added a check to ensure msg.content is not None
            elif hasattr(msg, "content") and msg.content is not None:
                # Handle LCEL message objects
                content = str(msg.content).strip()
                if content:
                    filtered_messages.append(msg)
            elif isinstance(msg, dict):
                # Handle dict-style messages
                content = str(msg.get("content", "")).strip()
                if content:
                    filtered_messages.append(msg)
            else:
                # This will now correctly skip messages where content is None
                logger.debug(f"Skipping message of unknown type or with None content: {type(msg)}")
        
        return filtered_messages

    def _is_json(self, text):
        """Check if text is valid JSON"""
        try:
            json.loads(text)
            return True
        except (ValueError, TypeError):
            return False

    def _handle_interrupt(self, command_exception, streamlit_output=None):
        """Handle human-in-the-loop interrupts. If streamlit_output is provided, use it for output."""
        try:
            query = command_exception.data.get('query', 'Assistance needed')
            response = f"\n[🤝 Human Assistance Needed] {query}\nPlease provide your response:"
            if streamlit_output:
                streamlit_output.write(response)
                # In Streamlit, we can't resume, so just display
            else:
                print(f"\n[🤝 Human Assistance Needed] {query}")
                human_input = input("Your response: ").strip()
                if not human_input:
                    human_input = "No response provided"
                command_exception.resume({"data": human_input})
                logger.info("Human assistance provided, resuming conversation")
        except Exception as e:
            logger.error(f"Error handling interrupt: {e}")
            try:
                error_msg = "Unable to get human assistance"
                if streamlit_output:
                    streamlit_output.write(error_msg)
                else:
                    print(error_msg)
                command_exception.resume({"data": error_msg})
            except:
                pass  # If resume fails, the conversation will end

    def _process_event_value(self, value):
        """Process event values and extract assistant responses"""
        if isinstance(value, tuple) and len(value) == 2:
            value = value[1]
        
        if isinstance(value, dict) and "messages" in value and value["messages"]:
            last_msg = value["messages"][-1]
            
            content = getattr(last_msg, "content", None)
            if content is None and isinstance(last_msg, dict):
                content = last_msg.get("content", "")
            
            if content and not self._is_json(str(content)):
                return content  # Return the assistant's message content
        return None

    def stream_conversation(self, user_input: str, thread_id: str = "default-thread", streamlit_output=None):
        """Stream conversation updates with improved error handling. If streamlit_output is provided, output to Streamlit."""
        config = {"configurable": {"thread_id": thread_id}}
        response = ""
        try:
            for event in self.graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config
            ):
                for value in event.values():
                    content = self._process_event_value(value)
                    if content and not self._is_json(str(content)):
                        if streamlit_output:
                            response += content + "\n"
                            streamlit_output.write(response)
                        else:
                            response = content  # Set response to the latest assistant content
            return response
        except Command as cmd:
            self._handle_interrupt(cmd, streamlit_output=streamlit_output)
        except Exception as e:
            logger.error(f"Error in stream_conversation: {e}")
            if streamlit_output:
                streamlit_output.write(f"❌ Error: {e}")
            else:
                print(f"❌ Error: {e}")

    def print_state_snapshot(self, thread_id: str = "default-thread"):
        """Print detailed state information for debugging"""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            snapshot = self.graph.get_state(config)
            print(f"\n--- 📊 State Snapshot (Thread: {thread_id}) ---")
            print(f"Values: {snapshot.values}")
            print(f"Next: {snapshot.next}")
            print(f"Created: {snapshot.created_at}")
            print(f"Tasks: {len(snapshot.tasks) if snapshot.tasks else 0}")
            print("--- End Snapshot ---\n")
            
        except Exception as e:
            logger.error(f"Error getting state snapshot: {e}")
            print(f"❌ Could not retrieve state: {e}")

    def run_interactive_session(self):
        """Run the interactive chat session"""
        thread_id = "default-thread"
        
        print("🚀 Conversational AI Agent Started!")
        print("Commands: 'quit'/'exit'/'q' to stop, 'state' to view current state")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == "state":
                    self.print_state_snapshot(thread_id=thread_id)
                    continue
                elif not user_input:
                    print("Please enter a message.")
                    continue
                
                self.stream_conversation(user_input, thread_id=thread_id)
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                print(f"❌ Unexpected error: {e}")
                print("Type 'quit' to exit or continue chatting...")

def main():
    """Main entry point"""
    try:
        agent = ConversationalAgent()
        agent.run_interactive_session()
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        print(f"❌ Failed to start: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())