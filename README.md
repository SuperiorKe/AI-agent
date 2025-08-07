# AI Research & Study Assistant – Conversational Chatbot

A modern, production-ready conversational AI assistant for content creation, research, and social media management.  
Built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), [Streamlit](https://streamlit.io/), and Google Gemini.  
Includes real-time web search (Tavily), web page browsing, and a beautiful chat UI.

---

## Features

- **Conversational AI**: Powered by Google Gemini, OpenAI, or Anthropic (configurable).
- **Modern Chat UI**: Streamlit web app with instant user bubbles and natural assistant responses.
- **Web Search**: Integrates Tavily for up-to-date information.
- **Web Page Browsing**: Reads and summarizes content from any URL.
- **Content Creation**: Generates LinkedIn posts, Twitter threads, and more.
- **Content Scheduling**: Simulate scheduling posts for later.
- **Extensible Tools**: Easily add or swap tools and LLMs.
- **Robust Error Handling**: Clean error messages and dev-friendly warnings.
- **Testing**: Includes a comprehensive test suite for all major features.
- **Environment-based Config**: All keys/settings via `.env` file.
- **Startup-Friendly**: Minimal, readable, and easy to extend.

---

## Quick Start

### Option 1: Docker (Recommended)

#### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd AI-agent
```

#### 2. Setup Environment
```sh
# Copy environment template
cp env.template .env

# Edit with your API keys
nano .env  # or use your preferred editor
```

#### 3. Run with Docker Compose
```sh
# Production mode
docker-compose up -d

# Development mode (with live reloading)
docker-compose -f docker-compose.dev.yml up -d
```

#### 4. Access the Application
Open your browser to: http://localhost:8501

📖 **For detailed Docker instructions, see [DOCKER_README.md](DOCKER_README.md)**

### Option 2: Local Development

#### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd AI-agent
```

#### 2. Create and Activate a Virtual Environment
```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

#### 4. Configure Environment Variables

Create a `.env` file in the project root:

```
# LLM configuration
CHATBOT_MODEL=openai:gpt-4             # or another supported model
CHATBOT_API_KEY=your-openai-api-key    # OpenAI, Anthropic, or Google API key

# Tavily Search
TAVILY_API_KEY=your-tavily-api-key     # Get from https://app.tavily.com/

# (Optional) LangSmith Tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=basic_chatbot
```

---

## Usage

### **Web App (Recommended)**

Launch the Streamlit chat UI:

```sh
streamlit run streamlit_app.py
```

- Open your browser to the local URL shown (usually http://localhost:8501).
- Type your message and press Enter.
- Your message appears instantly; the assistant’s response appears as soon as it’s ready.

### **Command-Line Chatbot**

```sh
python basic_chatbot.py
```

- Interact with the agent via the terminal.

---

## How the Chat UI Works

- **Instant Feedback:** Your message appears in the chat immediately.
- **Natural Flow:** The assistant’s response is generated in the background and appears in its own bubble.
- **No Delays:** No more waiting for both bubbles to appear at once.
- **Classic Chat Experience:** Just like modern messaging apps.

---

## Testing

Run the test suite to verify all features:

```sh
python test_basic_chatbot.py
```

- Tests cover LinkedIn post generation, Twitter threads, scheduling, web browsing, error handling, and end-to-end chat flow.

---

## Project Structure

```
AGENT/
  basic_chatbot.py         # Main chatbot logic and tools
  streamlit_app.py         # Streamlit web app UI
  test_basic_chatbot.py    # Automated tests for all features
  requirements.txt         # All dependencies
  README.md                # This file
  static/                  # (Optional) Static assets for UI
  templates/               # (Optional) HTML templates
```

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

- **Add More Tools:** Import and add new tools to the `tools` list in `basic_chatbot.py`.
- **Change LLM Provider:** Update `CHATBOT_MODEL` and `CHATBOT_API_KEY` in `.env`.
- **UI Customization:** Edit `streamlit_app.py` for new features or design tweaks.
- **Testing:** Add or modify tests in `test_basic_chatbot.py`.

---

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Tavily Search](https://app.tavily.com/)
- [Streamlit](https://streamlit.io/)
- [Google Gemini](https://ai.google.dev/)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
