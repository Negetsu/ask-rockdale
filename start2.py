import os
import re
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

print("🚀 Advanced RAG Document Uploader - Hybrid Chunking Strategy")
print("=" * 60)

class AdvancedDocumentProcessor:
    """
    A class to load, process, and chunk documents using a hybrid strategy
    to improve RAG performance on both broad and specific queries.
    """
    def __init__(self):
        # Using text-embedding-004 which outputs 768 dimensions to match Supabase setup
        # Alternative: models/text-embedding-004 also works
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
        )
        
        # Keywords that indicate important specific content for targeted chunking
        self.important_keywords = [
            'dog', 'pet', 'animal', 'leash', 'park', 'recreation',
            'fire', 'pit', 'burn', 'flame', 'outdoor',
            'vote', 'voting', 'election', 'ballot', 'register',
            'permit', 'license', 'application', 'fee',
            'noise', 'sound', 'quiet', 'disturb',
            'parking', 'vehicle', 'car', 'truck',
            'business', 'commercial', 'retail', 'restaurant',
            'zoning', 'residential', 'industrial',
            'tax', 'property', 'assessment', 'payment',
            'smoking', 'tobacco', 'cigarette', 'vaping',
            'alcohol', 'beer', 'wine', 'liquor',
            'chicken', 'livestock', 'farm', 'agriculture'
        ]

    def load_documents(self, data_folder="data"):
        """Loads all .docx and .pdf files from a specified folder."""
        print(f"\n📄 Loading documents from '{data_folder}' folder...")
        all_documents = []
        
        if not os.path.exists(data_folder):
            print(f"❌ Error: '{data_folder}' folder not found!")
            return []
            
        file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) 
                      if f.endswith(('.docx', '.pdf'))]

        if not file_paths:
            print(f"⚠️ No .docx or .pdf files found in '{data_folder}'")
            return []

        for file_path in file_paths:
            print(f"  - Loading {os.path.basename(file_path)}...")
            try:
                if file_path.endswith('.docx'):
                    loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                
                documents = loader.load()
                
                # Add enhanced metadata to each document
                for doc in documents:
                    doc.metadata.update({
                        'source': os.path.basename(file_path),
                        'file_type': file_path.split('.')[-1],
                        'doc_length': len(doc.page_content),
                        'word_count': len(doc.page_content.split())
                    })
                
                all_documents.extend(documents)
                print(f"    ✅ Loaded {len(documents)} elements.")
            
            except Exception as e:
                print(f"    ❌ Error loading {os.path.basename(file_path)}: {e}")

        return all_documents

    def create_enhanced_chunks(self, documents):
        """Applies a triple chunking strategy to the loaded documents."""
        print("\n🔧 Creating enhanced chunks with a hybrid strategy...")
        all_enhanced_chunks = []
        
        for i, doc in enumerate(documents):
            content = doc.page_content.strip()
            if len(content) < 50:  # Skip very short or empty content
                continue
            
            # Strategy 1: Original Chunks (for broad context)
            original_chunks = self.text_splitter.split_documents([doc])
            for chunk in original_chunks:
                chunk.metadata['chunk_type'] = 'original'
                chunk.metadata['strategy'] = 'broad_context'
            all_enhanced_chunks.extend(original_chunks)
            
            # Strategy 2: Keyword-Focused Chunks (for specific questions)
            keyword_chunks = self.create_keyword_focused_chunks(doc)
            all_enhanced_chunks.extend(keyword_chunks)
            
            # Strategy 3: QA-Style Chunks (for rules and regulations)
            qa_chunks = self.create_qa_chunks(doc)
            all_enhanced_chunks.extend(qa_chunks)
        
        print(f"  ✅ Created {len(all_enhanced_chunks)} total raw chunks from all strategies.")
        return all_enhanced_chunks

    def create_keyword_focused_chunks(self, doc):
        """Creates small, targeted chunks around specific keywords."""
        content = doc.page_content
        keyword_chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', content) # Split into sentences

        for keyword in self.important_keywords:
            for i, sentence in enumerate(sentences):
                if keyword.lower() in sentence.lower() and len(sentence.strip()) > 20:
                    # Create a context window of (previous + current + next) sentences
                    start_index = max(0, i - 1)
                    end_index = min(len(sentences), i + 2)
                    context_window = sentences[start_index:end_index]
                    chunk_content = " ".join(s.strip() for s in context_window)

                    if len(chunk_content) > 100: # Ensure chunk is substantial
                        new_chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                'chunk_type': 'keyword_focused',
                                'strategy': 'specific_query',
                                'focus_keyword': keyword
                            }
                        )
                        keyword_chunks.append(new_chunk)
        return keyword_chunks

    def create_qa_chunks(self, doc):
        """Creates chunks from sentences that match rule-based patterns."""
        content = doc.page_content
        qa_chunks = []
        rule_patterns = [
            r'(shall not|shall be|must|required|prohibited|allowed|permitted)[^.!?]*[.!?]',
            r'(it is unlawful|violation|penalty|fine)[^.!?]*[.!?]',
            r'(application for|permit required|license fee)[^.!?]*[.!?]',
            r'(hours of operation|open from|closed to)[^.!?]*[.!?]'
        ]
        
        for pattern in rule_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Get expanded context around the match
                start = max(0, match.start() - 150)
                end = min(len(content), match.end() + 150)
                context = content[start:end].strip()
                
                # FIXED: Improved sentence boundary cleanup
                # Find first sentence start
                first_sentence = re.search(r'^.*?[.!?]\s+', context)
                if first_sentence and first_sentence.start() > 0:
                    context = context[first_sentence.end():]
                
                # Find last sentence end
                last_sentence = re.search(r'[.!?](?=\s|$)', context)
                if last_sentence:
                    context = context[:last_sentence.end()]

                if len(context) > 100:
                    qa_chunk = Document(
                        page_content=context,
                        metadata={
                            **doc.metadata,
                            'chunk_type': 'qa_style',
                            'strategy': 'rule_based',
                            'rule_pattern': pattern[:50]  # Truncate pattern for metadata
                        }
                    )
                    qa_chunks.append(qa_chunk)
        return qa_chunks

    def deduplicate_chunks(self, chunks):
        """Removes duplicate or highly similar chunks."""
        print("\n🔄 Deduplicating chunks...")
        unique_chunks = []
        seen_content = set()
        
        for chunk in chunks:
            # Normalize and create a signature for comparison
            signature = re.sub(r'\s+', ' ', chunk.page_content.lower().strip())
            
            if signature not in seen_content:
                seen_content.add(signature)
                unique_chunks.append(chunk)
        
        print(f"  ✅ Removed {len(chunks) - len(unique_chunks)} duplicates. Total unique chunks: {len(unique_chunks)}")
        return unique_chunks

def upload_with_progress(chunks, embedding_model, supabase_client: Client, batch_size=50):
    """Uploads chunks to Supabase with progress tracking and error handling."""
    print(f"\n📤 Uploading {len(chunks)} chunks to Supabase...")
    
    # Optional: Clear existing data from the table
    try:
        print("  - 🗑️ Clearing existing documents in Supabase table...")
        # Delete all rows - works with any primary key type
        result = supabase_client.table("documents").select("id").execute()
        if result.data:
            supabase_client.table("documents").delete().neq('id', -1).execute()
        print("    ✅ Cleared successfully.")
    except Exception as e:
        print(f"    ⚠️ Could not clear table (it might be empty): {e}")
    
    successful_uploads = 0
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    # Initialize the vector store with the first batch
    if not chunks:
        print("  - No chunks to upload.")
        return 0
        
    try:
        print(f"  - 📦 Uploading Batch 1/{total_batches}...")
        vector_store = SupabaseVectorStore.from_documents(
            documents=chunks[:batch_size],
            embedding=embedding_model,
            client=supabase_client,
            table_name="documents",
            query_name="match_documents"
        )
        successful_uploads += len(chunks[:batch_size])
        print(f"    ✅ Success! ({successful_uploads}/{len(chunks)} total)")
        
        # Upload remaining batches with retry logic
        for i in range(batch_size, len(chunks), batch_size):
            batch_num = (i // batch_size) + 1
            batch = chunks[i:i + batch_size]
            print(f"  - 📦 Uploading Batch {batch_num}/{total_batches}...")
            
            # FIXED: Added retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vector_store.add_documents(batch)
                    successful_uploads += len(batch)
                    print(f"    ✅ Success! ({successful_uploads}/{len(chunks)} total)")
                    time.sleep(0.5)  # Avoid rate limiting
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"    ⚠️ Attempt {attempt + 1} failed, retrying... ({e})")
                        time.sleep(2)
                    else:
                        print(f"    ❌ Failed after {max_retries} attempts: {e}")

    except Exception as e:
        print(f"    ❌ An error occurred during upload: {e}")
        
    return successful_uploads

def test_retrieval(vector_store):
    """Tests the system's retrieval on specific, targeted questions."""
    print("\n🧪 Testing retrieval with specific questions...")
    test_questions = [
        "What are the rules for dogs in public parks?",
        "Can I have a fire pit in my backyard?",
        "How do I register to vote?",
        "What permits do I need to start a business?",
        "What are the noise ordinance rules for nighttime?",
        "Can I keep chickens in a residential area?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Query: \"{question}\"")
        try:
            results = vector_store.similarity_search(question, k=3)
            if results:
                print(f"  ✅ Found {len(results)} relevant chunks.")
                for i, res in enumerate(results):
                    print(f"    - Result {i+1} (Strategy: {res.metadata.get('strategy', 'unknown')} | Source: {res.metadata.get('source')}):")
                    print(f"      \"{res.page_content[:150].strip()}...\"")
            else:
                print("  ❌ No results found.")
        except Exception as e:
            print(f"  ❌ Error during test query: {e}")

def main():
    """Main function to run the entire document processing and uploading pipeline."""
    start_time = time.time()
    
    # Ensure Supabase environment variables are set
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env file.")
        return

    try:
        supabase_client = create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")
        return
        
    processor = AdvancedDocumentProcessor()
    
    # 1. Load documents
    raw_docs = processor.load_documents()
    if not raw_docs:
        print("❌ No documents loaded. Exiting.")
        return
    
    # 2. Create chunks using the hybrid strategy
    enhanced_chunks = processor.create_enhanced_chunks(raw_docs)
    
    # 3. Deduplicate the chunks
    unique_chunks = processor.deduplicate_chunks(enhanced_chunks)
    
    # 4. Upload to Supabase
    uploaded_count = upload_with_progress(
        unique_chunks, 
        processor.embedding_model, 
        supabase_client,
        batch_size=25  # FIXED: Reduced from 50 to avoid rate limits
    )
    
    end_time = time.time()
    print("\n" + "=" * 60)
    print(f"🎉 Pipeline finished in {end_time - start_time:.2f} seconds!")
    print(f"📊 Successfully uploaded {uploaded_count} unique chunks to the 'documents' table.")
    
    # 5. Test the system with a few queries
    if uploaded_count > 0:
        vector_store = SupabaseVectorStore(
            client=supabase_client,
            embedding=processor.embedding_model,
            table_name="documents",
            query_name="match_documents"
        )
        test_retrieval(vector_store)

    print("\n🏆 Enhanced RAG system is ready! Good luck with the Congressional App Challenge!")

if __name__ == "__main__":
    main()