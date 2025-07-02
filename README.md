# LangGraph Chatbot with Gemini, Tavily Search, and LangSmith Tracing

This project is a conversational AI chatbot built using [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and Google Gemini. It features web search integration via Tavily and supports tracing and monitoring with LangSmith. The chatbot is easily configurable for different LLM providers and tools using a `.env` file.

## Features
- **Conversational AI**: Powered by Google Gemini (or other supported LLMs)
- **Web Search**: Uses Tavily Search Engine for up-to-date information
- **Tracing & Monitoring**: Integrated with LangSmith for debugging and analytics
- **Easy Configuration**: All API keys and settings are managed via `.env`
- **Extensible**: Add more tools or swap LLMs with minimal code changes

---

## Setup

### 1. Clone the repository
```sh
# Example (adjust path as needed)
git clone <your-repo-url>
cd AGENT
```

### 2. Create and activate a virtual environment (recommended)
```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```sh
pip install -U langchain langgraph langchain-tavily langchain-google-genai python-dotenv
```

### 4. Set up your `.env` file
Create a `.env` file in the project root with the following content:

```
# LLM configuration
CHATBOT_MODEL=gemini-pro                # or another supported model
CHATBOT_API_KEY=your-gemini-api-key     # Gemini API key

# Tavily Search
TAVILY_API_KEY=your-tavily-api-key      # Get from https://app.tavily.com/

# LangSmith Tracing (optional but recommended)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=basic_chatbot
```

- Replace all `your-...-api-key` values with your actual API keys.
- You can use OpenAI or Anthropic by changing `CHATBOT_MODEL` and `CHATBOT_API_KEY` accordingly.

---

## Usage

Run the chatbot:
```sh
python basic_chatbot.py
```

Type your questions at the prompt. The chatbot will use Gemini for conversation and Tavily for web search when needed. All interactions are traced to LangSmith if tracing is enabled.

To exit, type `quit`, `exit`, or `q`.

---

## Environment Variables

| Variable              | Description                                      |
|----------------------|--------------------------------------------------|
| CHATBOT_MODEL        | LLM model string (e.g., `gemini-pro`, `openai:gpt-4.1`) |
| CHATBOT_API_KEY      | API key for the selected LLM provider             |
| TAVILY_API_KEY       | Tavily Search Engine API key                      |
| LANGSMITH_TRACING    | Set to `true` to enable LangSmith tracing         |
| LANGSMITH_ENDPOINT   | LangSmith API endpoint                            |
| LANGSMITH_API_KEY    | LangSmith API key                                 |
| LANGSMITH_PROJECT    | Project name for LangSmith traces                 |

---

## Extending the Chatbot
- **Add More Tools**: Import and add new tools to the `tools` list in `basic_chatbot.py`.
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
This project is for educational and research purposes. See LICENSE for details if provided. 

This is a test commit to verify the push functionality.