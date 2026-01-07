# utility imports
import tempfile
import time
from pathlib import Path

# Local file imports
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
# Ai imports
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import soundGen
from tools.pdf import loadPdf

# Load environment variables
load_dotenv()


class Agent:
    chatClient: ChatLlamaCpp
    vectorStore: Chroma
    embeddingClient: OllamaEmbeddings
    chromaPath: str

    class VectorSearchInput(BaseModel):
        query: str = Field(description="The search query to find relevant documents")
        k: int = Field(
            default=3,
            ge=1,
            le=10,
            description="Number of top similar chunks to retrieve (default: 3, max: 10)",
        )

    def __init__(self, template: str):
        # Local models - no API keys needed
        self.embeddingClient = OllamaEmbeddings(
            model="embeddinggemma:300m",
        )
        self.chatClient = ChatLlamaCpp(
            temperature=0.1,
            model_path="/home/oj2/Downloads/models/gemma-3-4b-it-Q3_K_S.gguf",
            max_tokens=256,
            n_ctx=1000,  # Reduced to prevent VRAM overflow
            n_gpu_layers=100,  # Increased for better GPU utilization (fits in 6GB VRAM)
            n_threads=4,  # Limited to prevent CPU competition with RAM
            verbose=False,
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
            tools=[self.vector_search_tool_instance]
        )

    def process_pdf(self, uploaded_file):
        """Process Streamlit UploadedFile into organized storage with metadata extraction"""

        # Extract original filename for fallback naming
        original_filename = uploaded_file.name
        safe_filename = (
            Path(original_filename).stem.replace("/", "_").replace("\\", "_")
        )

        # Create temp file
        with tempfile.NamedTemporaryFile(
            dir=self.tempDir, delete=False, suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Create loader
        loader = PyPDFLoader(tmp_file_path)

        # Process with enhanced metadata extraction
        # Pass OpenAI embedding function to ensure consistency
        loadPdf(loader, self.vectorStore)

        return loader

    def GenOpenAI(self, question: str):
        # NOTE: Reading system prompt from external file
        system_prompt: str

        with open("systemPrompt.md") as f:
            system_prompt = f.read()

        """Tool-calling agent that can dynamically search documents."""
        # System prompt for tool-calling agent
        system_prompt1 = """You are an academic research companion specializing in analyzing PDF documents.

CRITICAL: Always use the vector_search_tool FIRST when answering questions that could benefit from document evidence, even if you think you know the answer from training data. Document-specific information takes precedence over general knowledge.

Guidelines:
- For ANY question related to uploaded content: Use vector_search_tool first
- For questions requiring evidence or specific details: Use vector_search_tool first
- For general knowledge questions: Answer directly only if no documents are relevant
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
            # Add the assistant's tool-calling message to maintain proper conversation structure
            messages.append(response)

            # Execute tools and add results to messages
            for tool_call in response.tool_calls:
                if tool_call["name"] == "vector_search":
                    tool_result = self.vector_search_tool(
                        tool_call["args"]["query"], tool_call["args"].get("k", 3)
                    )
                    # Add tool result as a ToolMessage
                    messages.append(
                        ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                    )

            # Stream final response with tool results
            final_response_stream = self.tool_bound_chat_client.stream(messages)
            return final_response_stream

        # No tools called - stream direct response
        response_stream = self.chatClient.stream(messages)
        return response_stream

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

        @tool("vector_search", args_schema=self.VectorSearchInput)
        def vector_search_tool(query: str, k: int = 3) -> str:
            """Search the vector store for relevant document chunks based on the query."""
            return self.vector_search_tool(query, k)

        return vector_search_tool

    def monitor_cost(self, response) -> dict:
        """Monitor local model usage (no costs for local inference).

        Args:
            response: Local model response object

        Returns:
            dict: Usage information (local models have no costs)
        """
        return {
            "model": "qwen3-4b-local",
            "estimated_cost_cents": 0.0,  # No cost for local models
            "input_tokens": 0,  # Could extract from response if needed
            "output_tokens": 0,  # Could extract from response if needed
            "tool_calls": 0,  # Could count tool calls if needed
            "implemented": False,  # Local models don't need cost tracking
        }


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

    openaiAgent = Agent(template=sysTemplate)

    # Checking if chroma db is empty
    if openaiAgent.vectorStore._collection.count() == 0:
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
                response_stream = openaiAgent.GenOpenAI(userPrompt)
                # For streaming, we can't store the full response easily
                # Display it directly and optionally store final result
                st.write_stream(response_stream)

    # Display persisted answer
    if "answer" in st.session_state:
        st.write(st.session_state["answer"])

        # TTS button
        if st.button("Generate Audio"):
            openaiAgent.tts(st.session_state["answer"], "output.wav")
            st.session_state["audio_file"] = "output.wav"

    # Display persisted audio if available
    if "audio_file" in st.session_state:
        st.audio(st.session_state["audio_file"])

    if uploadedPdf:
        # PDF post processing
        if st.button("Process Pdf"):  # Ensuring button is clicked
            try:
                # Stage 1: File processing and permanent storage
                with st.spinner("📄 Processing uploaded PDF..."):
                    st.info(
                        "🔄 Step 1/2: Creating PDF loader and organizing storage..."
                    )
                    processed_pdf = openaiAgent.process_pdf(uploadedPdf)
                    time.sleep(0.5)  # Brief pause for UX

                # Stage 2: Document loading and vector store addition (now handled in process_pdf)
                with st.spinner("📚 Loading documents and adding to knowledge base..."):
                    st.info(
                        "🔄 Step 2/2: Processing and indexing content with OpenAI embeddings..."
                    )
                    # loadPdf is now called within process_pdf method with proper OpenAI embedding function
                    time.sleep(0.5)

                # Success notification
                st.success("✅ PDF processing complete!")
                st.balloons()  # Celebration animation

                # Show processing summary
                db_size = openaiAgent.vectorStore._collection.count()
                st.info(f"📊 Knowledge base now contains {db_size} document chunks")

            except Exception as e:
                st.error(f"❌ PDF processing failed: {str(e)}")
                st.warning("💡 Check the console for detailed error information")


if __name__ == "__main__":
    run_streamlit_app()
