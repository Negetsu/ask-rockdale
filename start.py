import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import UnstructuredWordDocumentLoader as UWDL
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

# Load API keys
load_dotenv(find_dotenv())
print("Loaded API keys from .env file.")

# Setup
data_folder = "data"
all_documents = []

# --- Load .docx files ---
docx_files = [f for f in os.listdir(data_folder) if f.endswith('.docx')]
for docx_file in docx_files:
    file_path = os.path.join(data_folder, docx_file)
    print(f"  > Loading DOCX: {file_path}...")
    loader = UWDL(file_path)
    documents = loader.load()
    all_documents.extend(documents)

# --- Load .pdf files ---
pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
for pdf_file in pdf_files:
    file_path = os.path.join(data_folder, pdf_file)
    print(f"  > Loading PDF: {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    all_documents.extend(documents)

# Summary
print("\n---------------------------------")
print(f"Total documents loaded: {len(all_documents)}")
print("---------------------------------")

# Preview first document
if all_documents:
    print("\nPreview of the first document:")
    print(all_documents[0].page_content[:20])

# Split documents into chunks
print("\nSplitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(all_documents)
print(f"\nSuccessfully split {len(all_documents)} pages into {len(chunks)} chunks.")

# Preview first chunk
if chunks:
    print("\n--- Preview of the first chunk: ---")
    print(chunks[0].page_content)
    print("\n--- Metadata of the first chunk: ---")
    print(chunks[0].metadata)

# Supabase setup
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_ANON_KEY in .env file")
supabase_client = create_client(supabase_url, supabase_key)

# Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Upload to Supabase
print("\n🚀 Uploading chunks to Supabase...")
vector_store = SupabaseVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    client=supabase_client,
    table_name="documents",
    query_name="match_documents"
)
print("✅ Upload complete!")