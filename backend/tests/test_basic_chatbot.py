import unittest
import os
from unittest.mock import patch
from chatbot import (
    generate_linkedin_post,
    generate_twitter_thread,
    schedule_content,
    browse_web_page,
    ConversationalAgent
)

class TestBasicChatbot(unittest.TestCase):
    def setUp(self):
        # Set up environment variables for dev mode
        os.environ["ENV"] = "dev"
        os.environ["CHATBOT_API_KEY"] = "fake-key"
        os.environ["TAVILY_API_KEY"] = "fake-key"

    def test_generate_linkedin_post(self):
        result = generate_linkedin_post.invoke({"topic": "AI in marketing"})
        self.assertIsInstance(result, str)
        self.assertTrue(len(result.strip()) > 0)

    def test_generate_twitter_thread(self):
        result = generate_twitter_thread.invoke({"topic": "AI in education", "num_tweets": 3})
        self.assertIsInstance(result, str)
        self.assertTrue(len(result.strip()) > 0)

    def test_schedule_content(self):
        content = "Test post"
        platform = "LinkedIn"
        dt = "2024-12-31 10:00"
        result = schedule_content.invoke({"content": content, "platform": platform, "datetime": dt})
        self.assertIn(platform, result)
        self.assertIn(content, result)
        self.assertIn(dt, result)

    def test_browse_web_page_valid(self):
        # Use a simple, fast-loading site
        result = browse_web_page.invoke({"url": "https://example.com"})
        self.assertIsInstance(result, str)
        self.assertTrue("Example Domain" in result or len(result) > 0)

    def test_browse_web_page_invalid(self):
        result = browse_web_page.invoke({"url": "not-a-url"})
        self.assertIn("Invalid URL", result)

    @patch.dict(os.environ, {"CHATBOT_API_KEY": "fake-key", "TAVILY_API_KEY": "fake-key", "ENV": "dev"})
    def test_missing_api_keys_dev_mode(self):
        # Should not raise in dev mode
        try:
            agent = ConversationalAgent()
        except Exception as e:
            self.fail(f"ConversationalAgent raised unexpectedly in dev mode: {e}")

    def test_end_to_end_chat(self):
        agent = ConversationalAgent()
        # Simulate a simple chat
        response = agent.stream_conversation("Generate a LinkedIn post about AI", thread_id="test-thread")
        print(f"End-to-end chat response: {response!r}")
        self.assertIsInstance(response, str)

if __name__ == "__main__":
    unittest.main() 