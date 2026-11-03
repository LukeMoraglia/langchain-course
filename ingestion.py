import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore



load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    FILE_PATH = "mediumblog1.txt"
    
    # Read the text file directly (no loader package required)
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        text_content = f.read()

    # Wrap it in a LangChain Document with clean, flat metadata
    document = [
        Document(
            page_content=text_content,
            metadata={"source": FILE_PATH}
        )
    ]

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    print("ingesting embeddings to pinecone...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("finished ingesting")
