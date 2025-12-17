# utility imports
import tempfile
from langchain_chroma import Chroma

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
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
            base_url="http://172.16.5.250:3000", model="embeddinggemma:300m"
        )
        self.chatClient = ChatOllama(
            base_url="http://172.16.5.250:3000", model="qwen3:4b"
        )
        self.chromaPath = "./Data/Chroma"
        self.vectorStore = Chroma(
            collection_name="pdf_collection",
            embedding_function=self.embeddingClient,
            persist_directory=self.chromaPath,  # Where to save data locally, remove if not necessary
        )
        self.tempDir = "./Data/deps/temp/"

        # TTS function
        self.tts = soundGen.text_to_speech

        # Create and bind the vector search tool
        self.vector_search_tool_instance = self.create_vector_search_tool()
        self.tool_bound_chat_client = self.chatClient.bind_tools(
            [self.vector_search_tool_instance]
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
        """Tool-calling agent that can dynamically search documents."""
        # System prompt for tool-calling agent
        system_prompt = """You are an academic research companion specializing in analyzing PDF documents.

You have access to a vector_search_tool that can search through uploaded documents.
Use this tool when you need specific information from the documents.

Guidelines:
- For questions requiring document evidence: Use vector_search_tool first
- For general knowledge questions: Answer directly from your training
- Always cite document sources when using search results
- If no relevant documents exist, clearly state this

Be concise, academic, and evidence-based in your responses."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]

        # Get initial LLM response
        response = self.tool_bound_chat_client.invoke(messages)

        # Check if LLM wants to call tools
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Execute tools and add results to messages
            for tool_call in response.tool_calls:
                if tool_call["name"] == "vector_search_tool":
                    tool_result = self.vector_search_tool(
                        tool_call["args"]["query"], tool_call["args"].get("k", 3)
                    )
                    # Add tool result as a ToolMessage
                    messages.append(
                        ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                    )

            # Get final response with tool results
            final_response = self.tool_bound_chat_client.invoke(messages)
            return final_response

        # No tools called - return direct response
        return response

    def is_db_empty(self) -> bool:
        return self.vectorStore._collection.count() == 0

    def vector_search_tool(self, query: str, k: int = 3) -> str:
        """Search the vector store for relevant document chunks based on the query.

        Args:
            query: The search query to find relevant documents
            k: Number of top similar chunks to retrieve (default: 3, max: 10)

        Returns:
            Formatted string of relevant document chunks with metadata
        """
        try:
            # Check if vector store has content
            if self.vectorStore._collection.count() == 0:
                return "No documents found in the knowledge base. Please upload and process some PDFs first."

            # Perform similarity search using existing retriever pattern
            retriever = self.vectorStore.as_retriever(search_kwargs={"k": min(k, 10)})
            docs = retriever.invoke(query)

            if not docs:
                return "No relevant documents found for the query."

            # Format results with metadata (similar to current PDF processing)
            formatted_results = []
            for i, doc in enumerate(docs, 1):
                page_label = doc.metadata.get("page_label", "Unknown")
                source = doc.metadata.get("source", "Unknown")
                content = doc.page_content

                formatted_results.append(
                    f"Document {i}:\n"
                    f"Source: {source}\n"
                    f"Page: {page_label}\n"
                    f"Content: {content}\n"
                    f"{'-' * 50}"
                )

            return "\n\n".join(formatted_results)

        except Exception as e:
            return f"Error performing vector search: {str(e)}"

    def create_vector_search_tool(self):
        """Create the vector search tool for binding to the LLM."""

        @tool
        def vector_search_tool(query: str, k: int = 3) -> str:
            """Search the vector store for relevant document chunks based on the query."""
            return self.vector_search_tool(query, k)

        return vector_search_tool


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

    #Checking if chroma db is empty
    if ollamaAgent.vectorStore._collection.count() == 0:
        st.info("The vector store is empty, upload something", icon="ℹ️")
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
                time.sleep(0.5)

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
