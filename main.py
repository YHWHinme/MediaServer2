#utility imports
from langchain.agents import create_agent
from langchain_core import embeddings
from langchain_chroma import Chroma

# Ai imports
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Local file imports
from tools.pdf import loadPdf
from tools.video import extract_frames
import streamlit as st


class Agent:
    chatClient: ChatOllama
    vectorStore: Chroma
    embeddingClient: OllamaEmbeddings
    chromaPath: str

    def __init__(self, systemPrompt:str):
        self.embeddingClient = OllamaEmbeddings(
            base_url="http://172.16.5.60:3000", model="embeddinggemma:300m"
        )
        self.chatClient = ChatOllama(
            base_url="http://172.16.5.60:3000", model="gemma3:270m"
        )
        self.vectorStore = Chroma(
            collection_name="pdf_collection",
            embedding_function=self.embeddingClient,
            persist_directory= self.chromaPath,  # Where to save data locally, remove if not necessary
        )
        self.chromaPath = "./Data/Chroma"
        # Import functions into the class
        self.systemPrompt = systemPrompt

        def GenOllama(self, chat_prompt: str):
            pass

        def checkEmpty(self) -> bool | bool:
            if self.vectorStore._collection.count(): 
                return False
            else:
                return True




def main():
    sysPrompt= """
    Some chat_prompt
    """
    ollamaAgent = Agent(sysPrompt)

    chat_prompt = st.chat_input(accept_file=True, placeholder="Start the messaging")
    if chat_prompt:
        with st.chat_message('user'):
            st.write_stream()



if __name__ == "__main__":
    main()
