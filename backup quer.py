import os
import dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
dotenv.load_dotenv()
# 1. Initialize the Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase_client = create_client(supabase_url, supabase_key)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 3. Connect to your existing Supabase vector store
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embedding_model,
    table_name="documents",
    query_name="match_documents"
)

query = "What are the rules for dogs in public parks?"
relevant_chunks = vector_store.similarity_search(query, k=4)

print(f"--- Found {len(relevant_chunks)} relevant chunks for the query: '{query}' ---")
for i, chunk in enumerate(relevant_chunks):
    print(f"\n--- Chunk {i+1} (Source: {chunk.metadata.get('source')}) ---")
    print(chunk.page_content)
#test gwohfowhefowhofhweoiufhwoe