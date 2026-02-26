# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Python-based AI Content Assistant chatbot built with LangGraph, LangChain, and Google Gemini. Single backend service in `backend/`. No frontend (Streamlit UI referenced in README does not exist in the repo).

### Environment setup

- Python 3.12 virtual environment lives at `backend/venv/`
- Always activate with `source /workspace/backend/venv/bin/activate` before running anything
- Missing from `requirements.txt` but required at runtime: `python-dotenv`, `pytest`; these are installed by the update script

### Running in dev mode

Set `ENV=dev` to bypass API key validation errors. Without real API keys, the chatbot initializes but cannot make LLM calls.

```sh
cd /workspace/backend
source venv/bin/activate
ENV=dev CHATBOT_API_KEY=fake-key TAVILY_API_KEY=fake-key python -c "from chatbot import ConversationalAgent; ConversationalAgent()"
```

### Running tests

```sh
cd /workspace/backend
source venv/bin/activate
ENV=dev CHATBOT_API_KEY=fake-key TAVILY_API_KEY=fake-key python -m pytest tests/test_basic_chatbot.py -v
```

- 4 of 6 tests pass without real API keys (`schedule_content`, `browse_web_page_valid`, `browse_web_page_invalid`, `missing_api_keys_dev_mode`)
- 2 tests (`test_generate_linkedin_post`, `test_generate_twitter_thread`) require valid `CHATBOT_API_KEY` (Google Gemini) to pass

### Required secrets for full functionality

| Secret | Purpose |
|---|---|
| `CHATBOT_API_KEY` | Google Gemini (or OpenAI/Anthropic) API key for LLM calls |
| `TAVILY_API_KEY` | Tavily search API key for web search tool |

### Gotchas

- The `chatbot.py` imports from `constitution_utils` using a bare module name; tests and scripts must be run from the `backend/` directory (or with `backend/` on `PYTHONPATH`)
- `browse_web_page` tool restricts domains to `example.com` and `wikipedia.org` for security; HTTPS requests may fail in sandbox environments due to SSL certificate verification issues
- No linter is configured in the repo; there is no `pyproject.toml`, `setup.cfg`, or linting config file
- `test_content_creation.py` is a manual test script with all test calls commented out; it is not runnable as an automated test
