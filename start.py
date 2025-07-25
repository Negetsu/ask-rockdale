import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader as UWDL
from langchain_text_splitters import RecursiveCharacterTextSplitter
#data
all_documents = []
data_folder = "data"
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

text_splitter = RecursiveCharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([all_documents])
print(texts[0])