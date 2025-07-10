# Conversational AI Chatbot (LangGraph, Gemini, Tavily, LangSmith)

A production-ready, extensible conversational AI chatbot built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and Google Gemini. Features include real-time web search via Tavily and optional tracing/monitoring with LangSmith. Easily configurable for different LLM providers and tools using environment variables.

---

## Features
- **Conversational AI**: Powered by Google Gemini or other supported LLMs
- **Web Search**: Integrates Tavily for up-to-date information
- **Web Page Browsing**: Can read and summarize the content of specific URLs.
- **Tracing & Monitoring**: Optional LangSmith integration for debugging and analytics
- **Configurable**: All API keys and settings managed via `.env`
- **Extensible**: Add tools or swap LLMs with minimal code changes

---

## Quick Start

### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd AGENT
```

### 2. Create and Activate a Virtual Environment (Recommended)
```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:

```
# LLM configuration
CHATBOT_MODEL=gemini-pro                # or another supported model
CHATBOT_API_KEY=your-gemini-api-key     # Gemini, OpenAI, or Anthropic API key

# Tavily Search
TAVILY_API_KEY=your-tavily-api-key      # Get from https://app.tavily.com/

# LangSmith Tracing (optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=basic_chatbot
```

- Replace all `your-...-api-key` values with your actual API keys.
- To use OpenAI or Anthropic, update `CHATBOT_MODEL` and `CHATBOT_API_KEY` accordingly.

---

## Usage

Run the chatbot interactively:
```sh
python basic_chatbot.py
```

- Type your questions at the prompt.
- The chatbot uses Gemini for conversation, Tavily for web search, and can browse specific URLs if you provide them.
- If LangSmith tracing is enabled, all interactions are logged for analytics.
- To exit, type `quit`, `exit`, or `q`.

---

## Environment Variables

| Variable            | Description                                         |
|---------------------|-----------------------------------------------------|
| CHATBOT_MODEL       | LLM model string (e.g., `gemini-pro`, `openai:gpt-4.1`) |
| CHATBOT_API_KEY     | API key for the selected LLM provider               |
| TAVILY_API_KEY      | Tavily Search Engine API key                        |
| LANGSMITH_TRACING   | Set to `true` to enable LangSmith tracing           |
| LANGSMITH_ENDPOINT  | LangSmith API endpoint                              |
| LANGSMITH_API_KEY   | LangSmith API key                                   |
| LANGSMITH_PROJECT   | Project name for LangSmith traces                   |

---

## Extending the Chatbot
- **Add More Tools**: Import and add new tools to the `tools` list in `basic_chatbot.py`. For example, the `browse_web_page` tool was added to allow the agent to read content from URLs.
- **Change LLM Provider**: Update `CHATBOT_MODEL` and `CHATBOT_API_KEY` in `.env`.
- **Tracing**: View traces and analytics in your [LangSmith dashboard](https://smith.langchain.com/).

---

## Resources
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Tavily Search](https://app.tavily.com/)
- [LangSmith](https://smith.langchain.com/)
- [Google Gemini](https://ai.google.dev/)

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
