import streamlit as st
from basic_chatbot import ConversationalAgent

# Initialize chatbot
if 'agent' not in st.session_state:
    st.session_state.agent = ConversationalAgent()

# Set page config
st.set_page_config(
    page_title="AI Research & Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to hide Streamlit UI elements
st.markdown("""
<style>
    /* Hide the top-right corner elements (Fork, GitHub, etc.) */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide the account menu at bottom */
    .stDeployButton {display: none;}
    
    /* Hide Streamlit logo */
    .stApp > header {display: none;}
    
    /* Hide the hamburger menu */
    .stApp > div[data-testid="stToolbar"] {display: none;}
    
    /* Hide the "made with streamlit" footer */
    .stApp > footer {display: none;}
    
    /* Additional hiding for deployment elements */
    .stApp > div[data-testid="stDecoration"] {display: none;}
    
    /* Ensure clean layout */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI Research & Study Assistant. I can help you with academic research, literature reviews, data analysis, study planning, and scholarly writing. What would you like to research or study today?"}
    ]

# Initialize awaiting_response flag
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

# Sidebar
with st.sidebar:
    st.title("📚 Research & Study Assistant")
    st.success(
        "A powerful AI assistant for academic research, literature reviews, study planning, and scholarly writing."
    )
    
    # Quick actions
    st.header("🔬 Research Tools")
    
    def sidebar_quick_actions():
        # Research topic input
        research_topic = st.text_input("Research Topic:", placeholder="Enter your research topic...")
        
        # Academic research tools
        if st.button("📖 Literature Review"):
            if research_topic:
                st.session_state.messages.append({"role": "user", "content": f"Help me conduct a literature review on {research_topic}. Include recent studies, key findings, and research gaps."})
                st.session_state.awaiting_response = True
                st.rerun()
        
        if st.button("📊 Research Methodology"):
            if research_topic:
                st.session_state.messages.append({"role": "user", "content": f"Suggest appropriate research methodologies for studying {research_topic}. Include quantitative and qualitative approaches."})
                st.session_state.awaiting_response = True
                st.rerun()
        
        if st.button("📝 Academic Writing"):
            if research_topic:
                st.session_state.messages.append({"role": "user", "content": f"Help me write an academic paper introduction about {research_topic}. Include background, problem statement, and objectives."})
                st.session_state.awaiting_response = True
                st.rerun()
        
        # Study assistance tools
        st.header("📚 Study Assistance")
        study_subject = st.text_input("Study Subject:", placeholder="What are you studying?")
        
        if st.button("🎯 Study Plan"):
            if study_subject:
                st.session_state.messages.append({"role": "user", "content": f"Create a comprehensive study plan for {study_subject}. Include learning objectives, timeline, and study strategies."})
                st.session_state.awaiting_response = True
                st.rerun()
        
        if st.button("❓ Practice Questions"):
            if study_subject:
                st.session_state.messages.append({"role": "user", "content": f"Generate practice questions and problems for {study_subject} to test my understanding."})
                st.session_state.awaiting_response = True
                st.rerun()
        
        if st.button("📋 Summary & Notes"):
            if study_subject:
                st.session_state.messages.append({"role": "user", "content": f"Create a comprehensive summary and study notes for {study_subject} with key concepts and examples."})
                st.session_state.awaiting_response = True
                st.rerun()

        # Creator credit below input field
        st.markdown("""
        <div style="text-align: center; margin-top: 1rem; margin-bottom: 2rem;">
            <p style="color: #666; font-size: 0.9rem;">
                Built with ❤️ by <a href="https://johnny-dev.onrender.com/" target="_blank" style="color: #667eea; text-decoration: none; font-weight: bold;">John Ndelembi</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

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
st.title("📚 AI Research & Study Assistant")

# Display chat messages in a more chat-like format
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown("<div style='display: flex; align-items: center;'>\U0001F916 <div style='margin-left: 8px;'>" + message["content"] + "</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='display: flex; align-items: center;'>\U0001F464 <div style='margin-left: 8px; font-weight: bold; color: #0066cc;'>" + message["content"] + "</div></div>", unsafe_allow_html=True)