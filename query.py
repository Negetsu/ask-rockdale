import os
import dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables from .env
dotenv.load_dotenv()

# --- 1. INITIALIZE MODELS AND DATABASE ---
print("Initializing models and database connection...")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase_client = create_client(supabase_url, supabase_key)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize the Supabase vector store to connect to our existing data
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embedding_model,
    table_name="documents",
    query_name="match_documents"
)

# Initialize the Gemini chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
print("Initialization complete.")

# --- 2. CREATE THE RAG CHAIN ---

# This is our instruction template for the AI
# It tells the AI how to use the retrieved documents ('context') to answer the question.
prompt_template = """
You are an assistant for citizens of Rockdale County. Your purpose is to answer questions based on the county's official documents.
Answer the user's question based only on the following context.
If the answer is not in the context, reply "I could not find the answer in the provided documents."

Context:
{context}

Question:
{input}

Answer:
"""

prompt = PromptTemplate.from_template(prompt_template)

# This chain takes the question and the retrieved documents and formats them into the prompt
document_chain = create_stuff_documents_chain(llm, prompt)

# This chain takes the user's question, gets the relevant documents, and passes them to the document_chain
retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), document_chain)


# --- 3. ASK A QUESTION ---
query = "Can I keep chickens in my backyard if I live in a residential area?"
print(f"\nAsking question: {query}")

# Invoke the chain. This runs the whole process.
response = retrieval_chain.invoke({"input": query})

# Print the final answer
print("\n--- Answer ---")
print(response["answer"])