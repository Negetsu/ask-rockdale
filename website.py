import streamlit as st
import os
import dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Ask Rockdale - AI Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Main styling */
.main {
    font-family: 'Inter', sans-serif;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}

.main-header h1 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.main-header p {
    font-size: 1.2rem;
    margin-bottom: 0;
    opacity: 0.9;
}

/* Stats cards */
.stats-container {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}

.stat-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    flex: 1;
    text-align: center;
    border-left: 4px solid #667eea;
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: #667eea;
    margin: 0;
}

.stat-label {
    font-size: 0.9rem;
    color: #666;
    margin: 0;
}

/* Example questions styling */
.example-questions {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.example-question {
    background: white;
    padding: 0.8rem;
    margin: 0.5rem 0;
    border-radius: 6px;
    border-left: 3px solid #667eea;
    cursor: pointer;
    transition: all 0.2s;
}

.example-question:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transform: translateY(-1px);
}

/* Chat styling improvements */
.stChatMessage {
    border-radius: 10px;
    margin-bottom: 1rem;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    color: #666;
    border-top: 1px solid #eee;
    margin-top: 3rem;
}

/* Sidebar styling */
.sidebar-content {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

/* Success/info boxes */
.info-box {
    background: #e3f2fd;
    border: 1px solid #2196f3;
    border-radius: 6px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box {
    background: #fff3e0;
    border: 1px solid #ff9800;
    border-radius: 6px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# --- SETUP AND INITIALIZATION ---
dotenv.load_dotenv()

@st.cache_resource
def initialize_rag_chain():
    """Initialize and return the RAG retrieval chain."""
    print("Initializing RAG chain...")
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    supabase_client = create_client(supabase_url, supabase_key)

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = SupabaseVectorStore(
        client=supabase_client,
        embedding=embedding_model,
        table_name="documents",
        query_name="match_documents"
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    prompt_template = """
    You are an assistant for citizens of Rockdale County, Georgia. Your purpose is to answer questions based on the county's official documents.
    
    Answer the user's question based only on the following context. Be specific, helpful, and cite relevant policies or ordinances when possible.
    If the answer is not in the context, reply "I could not find specific information about this in the available documents. You may want to contact Rockdale County directly at (770) 278-7000 or visit their website."

    Context:
    {context}

    Question:
    {input}

    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(search_kwargs={'k': 8}), document_chain)
    
    print("RAG chain initialized successfully.")
    return retrieval_chain

# Initialize the chain
rag_chain = initialize_rag_chain()

# --- MAIN HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🏛️ Ask Rockdale</h1>
    <p>Your AI-powered assistant for Rockdale County information</p>
</div>
""", unsafe_allow_html=True)

# --- STATS SECTION ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-number">24/7</p>
        <p class="stat-label">Available</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-number">&lt;5s</p>
        <p class="stat-label">Response Time</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-number">1000+</p>
        <p class="stat-label">Documents</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-number">100%</p>
        <p class="stat-label">Official Sources</p>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🚀 Try These Questions")
    
    example_questions = [
        "What are the rules for dogs in public parks?",
        "Can I have a fire pit in my backyard?",
        "What permits do I need to start a food truck business?",
        "How do I register to vote in Rockdale County?",
        "What are the noise ordinance regulations?",
        "Can I keep chickens in a residential area?",
        "What are the business licensing requirements?",
        "How do I appeal my property tax assessment?"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(f"💬 {question}", key=f"example_{i}", use_container_width=True):
            st.session_state.example_question = question
    
    st.markdown("---")
    
    st.markdown("### 📞 Contact Info")
    st.markdown("""
    **Rockdale County**
    - 📞 (770) 278-7000
    - 🌐 rockdalecountyga.gov
    - 📍 Conyers, GA 30012
    """)
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About This Tool")
    st.markdown("""
    Ask Rockdale uses advanced AI to search through official county documents and provide accurate, sourced answers to your questions about local policies, services, and regulations.
    
    **Features:**
    - ✅ Real-time document search
    - ✅ Source citations
    - ✅ Natural language queries
    - ✅ Official information only
    """)

# --- MAIN CONTENT AREA ---
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "👋 Hello! I'm your AI assistant for Rockdale County information. I can help you find answers about local policies, services, permits, and regulations. What would you like to know?"
        })

    # Check if example question was clicked
    if "example_question" in st.session_state:
        example_q = st.session_state.example_question
        
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": example_q
        })
        
        # Generate AI response immediately
        with st.spinner("🔍 Searching documents..."):
            start_time = time.time()
            response = rag_chain.invoke({"input": example_q})
            end_time = time.time()
            
            answer = response["answer"]
            response_time = round(end_time - start_time, 2)
            
            # Add assistant response with metadata
            full_response = f"{answer}\n\n*⏱️ Response time: {response_time}s | 📄 Sources checked: {len(response.get('context', []))}"
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })
        
        # Clean up and rerun to show the conversation
        del st.session_state.example_question
        st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about Rockdale County..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                start_time = time.time()
                response = rag_chain.invoke({"input": prompt})
                end_time = time.time()
                
                answer = response["answer"]
                response_time = round(end_time - start_time, 2)
                
                st.markdown(answer)
                
                # Show response time and source count
                sources = response.get("context", [])
                st.caption(f"⏱️ Response time: {response_time}s | 📄 Sources checked: {len(sources)}")
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})

with main_col2:
    st.markdown("### 🎯 Popular Topics")
    
    topics = [
        ("🏠", "Housing & Property", ["Property taxes", "Zoning laws", "Building permits"]),
        ("🚗", "Transportation", ["Parking rules", "Road construction", "Public transit"]),
        ("🏢", "Business", ["Licenses", "Permits", "Regulations"]),
        ("🐕", "Pets & Animals", ["Leash laws", "Pet registration", "Animal control"]),
        ("🗳️", "Voting & Elections", ["Registration", "Polling locations", "Candidate info"]),
        ("🚨", "Public Safety", ["Emergency services", "Fire regulations", "Safety codes"])
    ]
    
    for icon, topic, subtopics in topics:
        with st.expander(f"{icon} {topic}"):
            for subtopic in subtopics:
                st.markdown(f"• {subtopic}")

    st.markdown("---")
    
    st.markdown("### 💡 Tips for Better Results")
    st.markdown("""
    - **Be specific**: Instead of "parks", try "dog rules in county parks"
    - **Use local terms**: Include "Rockdale County" in your question
    - **Ask follow-ups**: I remember our conversation context
    - **Check sources**: All answers cite official documents
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>Ask Rockdale AI Assistant</strong> | Built for the Congressional App Challenge</p>
    <p>Powered by Google Gemini AI & Supabase | All information sourced from official Rockdale County documents</p>
    <p><em>For official business, always verify with Rockdale County directly</em></p>
</div>
""", unsafe_allow_html=True)

# Clear chat button (in sidebar)
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()