# utility imports
import tempfile
import os
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader

# Ai imports
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Local file imports
import streamlit as st
import time

from tools.pdf import loadPdf
import soundGen


class Agent:
    chatClient: ChatOllama
    vectorStore: Chroma
    embeddingClient: OllamaEmbeddings
    chromaPath: str

    def __init__(self, template: str):
        self.embeddingClient = OllamaEmbeddings(
            base_url="http://172.16.5.234:3000", model="embeddinggemma:300m"
        )
        self.chatClient = ChatOllama(
            base_url="http://172.16.5.234:3000", model="qwen3:4b"
        )
        self.chromaPath = "./Data/Chroma"
        self.vectorStore = Chroma(
            collection_name="pdf_collection",
            embedding_function=self.embeddingClient,
            persist_directory=self.chromaPath,  # Where to save data locally, remove if not necessary
        )
        self.tempDir = "./Data/deps/temp/"

        # New: Prompt template for academic PDF analysis
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an academic research companion specializing in analyzing PDF documents. Your role is to provide accurate, evidence-based answers using the provided context from academic sources.

Context from documents:
{context}

Instructions:
- Answer based solely on the provided context.
- Cite sources by referencing document metadata (e.g., page numbers or titles) if available in the context.
- If the context is insufficient, state "Insufficient information in the provided documents" and suggest rephrasing the question.
- Maintain an academic tone: objective, formal, and concise.
- Structure your response with headings like "Summary," "Key Insights," or "Conclusion" for clarity.""",
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        # New: Retriever for vector store
        self.retriever = self.vectorStore.as_retriever(search_kwargs={"k": 3})

        # TTS function
        self.tts = soundGen.text_to_speech

        # New: RAG chain
        # TODO: Find a way to get rid of this function and add HumanMessage directly into the rag chain
        def format_question(question: str):
            return [HumanMessage(content=question)]

        self.rag_chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough() | format_question,
            }
            | self.prompt
            | self.chatClient
        )

    def process_pdf(self, uploaded_file):
        """Process Streamlit UploadedFile into PyPDFLoader using temp file"""

        # Create temporary file from uploaded file bytes
        with tempfile.NamedTemporaryFile(
            dir=self.tempDir, delete=False, suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Create loader with temp file path (cleanup moved to loadPdf)
        loader = PyPDFLoader(tmp_file_path)
        return loader

    def GenOllama(self, question: str):
        return self.rag_chain.invoke(question)

    def is_db_empty(self) -> bool:
        return self.vectorStore._collection.count() == 0


def run_streamlit_app():
    sysTemplate = """
You are an academic research companion specializing in analyzing PDF documents. Your role is to provide accurate, evidence-based answers using the provided context from academic sources.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based solely on the provided context.
- Answer if the question isn't academic, don't give an academic answer- Cite sources by referencing document metadata (e.g., page numbers or titles) if available in the context.
- If the context is insufficient, state "Insufficient information in the provided documents" and suggest rephrasing the question.
- Maintain an academic tone: objective, formal, and concise.
- Structure your response with headings like "Summary," "Key Insights," or "Conclusion" for clarity.
"""

    ollamaAgent = Agent(template=sysTemplate)

    # Streamlit UI
    uploadedPdf = st.file_uploader(
        "Upload a pdf please", type="pdf"
    )  # uploading the pdf

    userPrompt = st.text_area("Prompting")
    if userPrompt:
        answerBtn = st.button("Send prompt")
        if answerBtn:
            with st.spinner("Processing llm..."):
                answer = ollamaAgent.GenOllama(userPrompt).content
                st.session_state["answer"] = answer
                time.sleep(.5)

    # Display persisted answer
    if "answer" in st.session_state:
        st.write(st.session_state["answer"])

        # TTS button
        if st.button("Generate Audio"):
            ollamaAgent.tts(st.session_state["answer"], "output.wav")
            st.session_state["audio_file"] = "output.wav"

    # Display persisted audio if available
    if "audio_file" in st.session_state:
        st.audio(st.session_state["audio_file"])

    if uploadedPdf:
        # PDF post processing
        if st.button("Process Pdf"):  # Ensuring button is clicked
            try:
                # Stage 1: File processing
                with st.spinner("📄 Processing uploaded PDF..."):
                    st.info("🔄 Step 1/2: Creating PDF loader...")
                    processed_pdf = ollamaAgent.process_pdf(uploadedPdf)
                    time.sleep(0.5)  # Brief pause for UX

                # Stage 2: Document loading and vector store addition
                with st.spinner("📚 Loading documents and adding to knowledge base..."):
                    st.info("🔄 Step 2/2: Processing and indexing content...")
                    loadPdf(processed_pdf, ollamaAgent.vectorStore)
                    time.sleep(0.5)

                # Success notification
                st.success("✅ PDF processing complete!")
                st.balloons()  # Celebration animation

                # Show processing summary
                db_size = ollamaAgent.vectorStore._collection.count()
                st.info(f"📊 Knowledge base now contains {db_size} document chunks")

            except Exception as e:
                st.error(f"❌ PDF processing failed: {str(e)}")
                st.warning("💡 Check the console for detailed error information")


if __name__ == "__main__":
    run_streamlit_app()
