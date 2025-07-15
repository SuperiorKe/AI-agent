import streamlit as st
from basic_chatbot import ConversationalAgent

# Initialize chatbot
if 'agent' not in st.session_state:
    st.session_state.agent = ConversationalAgent()

# Set page config
st.set_page_config(
    page_title="AI Content Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! I'm your AI Content Assistant. Ready to help you create amazing content, research topics, and manage your social media. What would you like to work on?"}
    ]

# Initialize awaiting_response flag
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

# Sidebar
with st.sidebar:
    st.title("AI Content Assistant")
    st.info(
        "A powerful AI assistant that can help you create content, research topics, and manage your social media posts."
    )
    
    # Quick actions
    st.header("Quick Actions")
    
    def sidebar_quick_actions():
        topic = st.text_input("Content Topic:")
        if st.button("Generate LinkedIn Post"):
            if topic:
                st.session_state.messages.append({"role": "user", "content": f"Generate a LinkedIn post about {topic}"})
                st.session_state.awaiting_response = True
                st.rerun()
        if st.button("Generate Twitter Thread"):
            if topic:
                st.session_state.messages.append({"role": "user", "content": f"Generate a Twitter thread about {topic}"})
                st.session_state.awaiting_response = True
                st.rerun()
    sidebar_quick_actions()

# Chat input with improved styling (must be before chat history rendering)
prompt = st.chat_input("Type your message here...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.awaiting_response = True
    st.rerun()

# If awaiting_response, generate assistant response and rerun
if st.session_state.awaiting_response:
    # Find the last user message without a following assistant message
    user_msgs = [i for i, m in enumerate(st.session_state.messages) if m["role"] == "user"]
    assistant_msgs = [i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"]
    if user_msgs and (not assistant_msgs or user_msgs[-1] > assistant_msgs[-1]):
        last_user_msg = st.session_state.messages[user_msgs[-1]]["content"]
        try:
            response = st.session_state.agent.stream_conversation(
                last_user_msg,
                thread_id="streamlit-chat",
                streamlit_output=None
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
    st.session_state.awaiting_response = False
    st.rerun()

# Main content
st.title("\U0001F916 AI Content Assistant")

# Display chat messages in a more chat-like format
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown("<div style='display: flex; align-items: center;'>\U0001F916 <div style='margin-left: 8px;'>" + message["content"] + "</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='display: flex; align-items: center;'>\U0001F464 <div style='margin-left: 8px; font-weight: bold; color: #0066cc;'>" + message["content"] + "</div></div>", unsafe_allow_html=True)