from langchain_core import embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
import tools

def main():
    class Agent:
        chatClient: ChatOllama
        ragClient: Chroma
        embeddingClient: OllamaEmbeddings
        # Tools half of the defination

        def __init__(self):
            chatClient = ChatOllama(
                base_url="http://172.16.5.60:3000",
                model="gemma3:270m"
            )
            ragClient = Chroma(
                collection_name="pdf_collection",
                embedding_function=self.embeddingClient,
                persist_directory="./Data/Chroma",  # Where to save data locally, remove if not necessary
            )
            embeddingClient = OllamaEmbeddings(
                base_url="http://172.16.5.60:3000",
                model="embeddinggemma:300m"
            )



if __name__ == "__main__":
    main()
