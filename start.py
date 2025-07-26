import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader as UWDL
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
#data
all_documents = []
data_folder = "data"
print("Loaded API keys from .env file.")
#docx stuff
docx_files = [f for f in os.listdir(data_folder) if f.endswith('.docx')]
for docx_file in docx_files:
    file_path = os.path.join(data_folder, docx_file)
    print(f"  > Loading {file_path}...")
    loader = UWDL(file_path)
    documents = loader.load()
    all_documents.extend(documents)
print("\n---------------------------------")
print(f"Total documents loaded: {len(all_documents)}")
print("---------------------------------")
#prints first few letters
if all_documents:
    print("\nPreview of the first document:")
    print(all_documents[0].page_content[:20])

# (Your document loading code from above is perfect)

print("\nSplitting documents into chunks...")

# Create an instance of the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

# Use the correct method: .split_documents()
chunks = text_splitter.split_documents(all_documents)

print(f"\nSuccessfully split {len(all_documents)} pages into {len(chunks)} chunks.")

# Let's inspect the first chunk
if chunks:
    print("\n--- Preview of the first chunk: ---")
    print(chunks[0].page_content)
    print("\n--- Metadata of the first chunk: ---")
    print(chunks[0].metadata)

# ai
#keys and make client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_ANON_KEY in .env file")
Googkey = os.getenv("GOOGLE_API_KEY")
supabase_client = create_client(supabase_url, supabase_key)
#model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = SupabaseVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    client=supabase_client,
    table_name="documents",
    query_name="match_documents"
)