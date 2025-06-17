import os
from dotenv import load_dotenv
from transformers import AutoModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("loading...")
    loader = TextLoader("/home/jawad/learning/projects/medium-rag/medium-rag/mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # Make sure the dimentionality matches with the Pinecone embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    print("ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("finish")