import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader as UWDL
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
print("✅ Loaded .env keys")

# Check Supabase credentials
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
Googkey = os.getenv("GOOGLE_API_KEY")

if not supabase_url or not supabase_key or not Googkey:
    raise ValueError("❌ Missing one or more required API keys in .env")

# Create Supabase client
supabase_client = create_client(supabase_url, supabase_key)
print("✅ Supabase client created")

# Load one .docx file
data_folder = "data"
docx_files = [f for f in os.listdir(data_folder) if f.endswith('.docx')]
if not docx_files:
    raise FileNotFoundError("❌ No .docx files found in data folder")

file_path = os.path.join(data_folder, docx_files[0])
print(f"📄 Loading file: {file_path}")
loader = UWDL(file_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} document(s)")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
chunks = splitter.split_documents(documents)
print(f"🔪 Split into {len(chunks)} chunks")

# Preview chunk
print("\n🔍 First chunk preview:")
print(chunks[0].page_content[:200])

# Create embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("✨ Created embedding model")

# Try inserting into Supabase
try:
    print("\n🚀 Uploading to Supabase...")
    SupabaseVectorStore.from_documents(
        documents=[chunks[0]],  # just the first one for debug
        embedding=embedding_model,
        client=supabase_client,
        table_name="documents",
        query_name="match_documents"
    )
    print("✅ Upload successful")
except Exception as e:
    print("❌ Upload failed:")
    print(e)
