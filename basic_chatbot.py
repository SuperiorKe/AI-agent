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
if not API_KEY:
    raise ValueError("CHATBOT_API_KEY environment variable is required")
if not TAVILY_API_KEY:
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
        tools = [tavily_search, human_assistance]

        # Initialize LLM with tools
        try:
            llm = init_chat_model(MODEL, model_provider=model_provider)
            llm_with_tools = llm.bind_tools(tools, enforce_single_tool=True)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

        # Create a system prompt to guide the LLM's tool usage
        system_prompt = (
            "You are a helpful assistant. You have access to a search tool and a special tool to request human assistance. "
            "If the user asks for 'expert guidance', 'human help', or explicitly asks you to 'request assistance', "
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

    def _handle_interrupt(self, command_exception):
        """Handle human-in-the-loop interrupts"""
        try:
            query = command_exception.data.get('query', 'Assistance needed')
            print(f"\n[🤝 Human Assistance Needed] {query}")
            
            human_input = input("Your response: ").strip()
            if not human_input:
                human_input = "No response provided"
            
            command_exception.resume({"data": human_input})
            logger.info("Human assistance provided, resuming conversation")
            
        except Exception as e:
            logger.error(f"Error handling interrupt: {e}")
            try:
                command_exception.resume({"data": "Unable to get human assistance"})
            except:
                pass  # If resume fails, the conversation will end

    def stream_conversation(self, user_input: str, thread_id: str = "default-thread"):
        """Stream conversation updates with improved error handling"""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            for event in self.graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config
            ):
                for value in event.values():
                    self._process_event_value(value)
                    
        except Command as cmd:
            self._handle_interrupt(cmd)
            
        except Exception as e:
            logger.error(f"Error in stream_conversation: {e}")
            print(f"❌ Error: {e}")

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
                print(f"🤖 Assistant: {content}")

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