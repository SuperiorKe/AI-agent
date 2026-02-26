# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

AI Content Assistant — a conversational chatbot CLI built with LangGraph, LangChain, and Google Gemini. All application code lives in `backend/`. See `README.md` for full feature list.

### Running the application

```bash
source venv/bin/activate
cd backend
ENV=dev python chatbot.py
```

Set `ENV=dev` to skip API-key validation errors. For full functionality, provide real keys via env vars or a `.env` file in the repo root:
- `CHATBOT_API_KEY` — Google Gemini (default), OpenAI, or Anthropic key
- `TAVILY_API_KEY` — Tavily web search key
- `CHATBOT_MODEL` — e.g. `gemini-pro`, `openai:gpt-4.1`, `anthropic:claude-3.5-sonnet`

### Running tests

```bash
source venv/bin/activate
cd backend
ENV=dev CHATBOT_API_KEY=fake-key TAVILY_API_KEY=fake-key python -m unittest tests.test_basic_chatbot -v
```

Two tests (`test_generate_linkedin_post`, `test_generate_twitter_thread`) require a valid `CHATBOT_API_KEY` because they invoke the LLM directly. The other 4 tests pass without real keys.

### Linting

```bash
source venv/bin/activate
ruff check backend/
```

The repo has no `ruff.toml` / `pyproject.toml` lint config. `ruff` is installed in the venv for convenience.

### Gotchas

- `requirements.txt` is missing `python-dotenv` and `requests` — they are installed separately in the update script.
- Tests must run from `backend/` because `test_basic_chatbot.py` imports `from chatbot import ...` (relative import).
- The README references `streamlit_app.py` and `basic_chatbot.py`, but the actual entry point is `backend/chatbot.py`.
- `browse_web_page` tool restricts domains to `example.com` and `wikipedia.org`.
- The cloud VM may have SSL certificate verification issues for outbound HTTPS; this causes `browse_web_page` tests to fail on network calls. This is an environment limitation, not a code bug.
