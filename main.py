#utility imports
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Ai imports
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Local file imports
import streamlit as st

from tools.pdf import loadPdf


class Agent:
    chatClient: ChatOllama
    vectorStore: Chroma
    embeddingClient: OllamaEmbeddings
    chromaPath: str

    def __init__(self, template: str):
        self.embeddingClient = OllamaEmbeddings(
            base_url="http://172.16.5.60:3000", model="embeddinggemma:300m"
        )
        self.chatClient = ChatOllama(
            base_url="http://172.16.5.60:3000", model="gemma3:270m"
        )
        self.chromaPath = "./Data/Chroma"
        self.vectorStore = Chroma(
            collection_name="pdf_collection",
            embedding_function=self.embeddingClient,
            persist_directory=self.chromaPath,  # Where to save data locally, remove if not necessary
        )

        # New: Prompt template for academic PDF analysis
        self.prompt = ChatPromptTemplate(template)

        # New: Retriever for vector store
        self.retriever = self.vectorStore.as_retriever(search_kwargs={"k": 3})

        # New: RAG chain
        self.rag_chain = {"context": self.retriever, "question": RunnablePassthrough()} | self.prompt | self.chatClient

    def GenOllama(self, question: str):
        return self.rag_chain.invoke(question)

    def is_db_empty(self) -> bool:
        return self.vectorStore._collection.count() == 0


def main():
    sysTemplate = """
You are an academic research companion specializing in analyzing PDF documents. Your role is to provide accurate, evidence-based answers using the provided context from academic sources.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based solely on the provided context.
- Cite sources by referencing document metadata (e.g., page numbers or titles) if available in the context.
- If the context is insufficient, state "Insufficient information in the provided documents" and suggest rephrasing the question.
- Maintain an academic tone: objective, formal, and concise.
- Structure your response with headings like "Summary," "Key Insights," or "Conclusion" for clarity.
"""

    ollamaAgent = Agent(template=sysTemplate)

    # Streamlit UI
    uploadedPdf = st.file_uploader("Upload a pdf please", type="pdf") #uploading the pdf
    st.write(uploadedPdf)
    # if uploadedPdf:
    #     button = st.button("Process Pdf") # Ensuring button is clicked
    #     if button: # Adding document to vector database if button is clicked
    #         loadPdf(uploadedPdf, ollamaAgent.vectorStore)



if __name__ == "__main__":
    main()
