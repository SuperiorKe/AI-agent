# AI Content Assistant – Conversational Chatbot

A modern, production-ready conversational AI assistant for content creation, research, and social media management.  
Built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and powered by major LLMs like Google Gemini, OpenAI, and Anthropic.  
Includes real-time web search (Tavily), secure web page browsing, and a robust command-line interface.

---

## Features

- **Conversational AI**: Powered by Google Gemini, OpenAI, or Anthropic (configurable).
- **Interactive Terminal UI**: Engage with the chatbot through a command-line interface.
- **Web Search**: Integrates Tavily for up-to-date information.
- **Secure Web Page Browsing**: Reads content from URLs with a security allowlist to prevent abuse.
- **Content Creation**: Generates LinkedIn posts and Twitter threads using platform-specific "constitutions" to guide tone and style.
- **Content Scheduling**: Simulates scheduling posts for later.
- **Extensible Tools**: Easily add or swap tools and LLMs.
- **Robust Error Handling**: Clean error messages and developer-friendly warnings.
- **Testing**: Includes a test suite covering core features.
- **Environment-based Config**: All keys and settings are managed via a `.env` file.

---

## Quick Start

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd <repo-name>
```

### 2. Create and Activate a Virtual Environment

```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

All required packages are listed in the `backend` directory.

```sh
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root by copying the example:

```sh
cp .env.example .env
```

Now, edit the `.env` file with your API keys:

```
# LLM configuration
CHATBOT_MODEL="gemini-pro"              # or "openai:gpt-4-turbo", "anthropic:claude-3-sonnet-20240229"
CHATBOT_API_KEY="your-llm-api-key"      # Gemini, OpenAI, or Anthropic API key

# Tavily Search
TAVILY_API_KEY="your-tavily-api-key"    # Get from https://app.tavily.com/

# (Optional) LangSmith Tracing
#LANGSMITH_TRACING="true"
#LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
#LANGSMITH_API_KEY="your-langsmith-api-key"
#LANGSMITH_PROJECT="ai-content-assistant"
```

---

## Usage

Run the chatbot from the root directory:

```sh
python backend/chatbot.py
```

- Interact with the agent directly in your terminal.
- Type `quit`, `exit`, or `q` to end the session.
- Type `state` to see a debug snapshot of the conversation state.

---

## Testing

Run the test suite from the root directory to verify all features:

```sh
python -m unittest discover backend/tests
```

- Tests cover content generation, tool usage, error handling, and core agent logic.

---

## Project Structure

```
.
├── backend/
│   ├── chatbot.py             # Main chatbot logic, tools, and interactive session
│   ├── constitution_utils.py  # Utilities for handling content constitutions
│   ├── constitutions/         # Constitution files for different platforms
│   ├── requirements.txt       # Python dependencies
│   └── tests/                 # Unit and integration tests
│       ├── test_basic_chatbot.py
│       └── test_content_creation.py
├── .env.example           # Example environment file
├── .gitignore
├── LICENSE
└── README.md
```

---

## Environment Variables

| Variable            | Description                                         |
|---------------------|-----------------------------------------------------|
| `CHATBOT_MODEL`       | LLM model string (e.g., `gemini-pro`, `openai:gpt-4o`) |
| `CHATBOT_API_KEY`     | API key for the selected LLM provider               |
| `TAVILY_API_KEY`      | Tavily Search Engine API key                        |
| `LANGSMITH_TRACING`   | Set to `true` to enable LangSmith tracing           |
| `LANGSMITH_ENDPOINT`  | LangSmith API endpoint                              |
| `LANGSMITH_API_KEY`   | LangSmith API key                                   |
| `LANGSMITH_PROJECT`   | Project name for LangSmith traces                   |

---

## Extending the Chatbot

- **Add More Tools:** Create new tool functions in `backend/chatbot.py` and add them to the `tools` list in the `ConversationalAgent` class.
- **Change LLM Provider:** Update `CHATBOT_MODEL` and `CHATBOT_API_KEY` in your `.env` file.
- **Add Constitutions:** Create a new `.txt` file in `backend/constitutions` to define guidelines for new content types.
- **Add Tests:** Create new test cases in the `backend/tests` directory to validate new functionality.

---

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Tavily Search](https://app.tavily.com/)
- [Google Gemini](https://ai.google.dev/)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
