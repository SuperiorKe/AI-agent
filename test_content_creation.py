# Manual test script for ConversationalAgent content creation tools.
# Note: This is not an automated test; it prints outputs for manual inspection.
from basic_chatbot import ConversationalAgent

def test_content_creation():
    agent = ConversationalAgent()
    
    # Test LinkedIn post generation
    print("\n=== Testing LinkedIn Post Generation ===")
    linkedin_prompt = """
    Generate a LinkedIn post about AI in marketing. 
    Use a professional style and include insights about recent trends.
    """
    try:
        agent.stream_conversation(linkedin_prompt)
    except Exception as e:
        print(f"Error: {e}")
    
    # Test Twitter thread generation
    print("\n=== Testing Twitter Thread Generation ===")
    twitter_prompt = """
    Create a 4-tweet thread about the future of AI in education.
    Make it engaging and include practical examples.
    """
    try:
        agent.stream_conversation(twitter_prompt)
    except Exception as e:
        print(f"Error: {e}")
    
    # Test content scheduling
    print("\n=== Testing Content Scheduling ===")
    schedule_prompt = """
    Schedule this LinkedIn post for tomorrow at 10 AM:
    'Exciting developments in AI are revolutionizing the marketing landscape...'
    """
    try:
        agent.stream_conversation(schedule_prompt)
    except Exception as e:
        print(f"Error: {e}")
    
    # Test combined workflow
    print("\n=== Testing Combined Workflow ===")
    combined_prompt = """
    1. Research recent trends in remote work
    2. Create a LinkedIn post about these trends
    3. Schedule the post for next Monday at 2 PM
    """
    try:
        agent.stream_conversation(combined_prompt)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_content_creation()
