# utility imports
import tempfile
import time
import os
from pathlib import Path

# Local file imports
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma

# Ai imports
from langchain_openai import ChatOpenAI
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

    def __init__(self):
        # Local models - no API keys needed
        self.embeddingClient = OllamaEmbeddings(
            model="embeddinggemma:300m",
        )
        self.chatClient = ChatOpenAI(
            api_key=os.getenv("ZEN_API_KEY"),
            base_url="https://opencode.ai/zen/v1",
            model="grok-code",
            temperature=0.1,
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

        # System prompt for tool-calling agent
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

            # Progress: Starting search
            yield from self._yield_progress_indicators(1, 3, "Analyzing search query")

            # Execute tools and add results to messages
            tool_result = None
            for tool_call in response.tool_calls:
                if tool_call["name"] == "vector_search":
                    tool_result = self.vector_search(
                        tool_call["args"]["query"], tool_call["args"].get("k", 3)
                    )
                    # Add tool result as a ToolMessage
                    messages.append(
                        ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                    )

            # Progress: Search complete, processing results
            yield from self._yield_progress_indicators(
                2, 3, "Executing document search"
            )

            # Extract document count for progress indicator
            # This is a bit hacky - we need to parse the tool result to get count
            doc_count = 0
            if tool_result and "Document 1:" in tool_result:
                # Count how many "Document X:" entries there are
                import re

                matches = re.findall(r"Document \d+:", tool_result)
                doc_count = len(matches)

            yield from self._yield_progress_indicators(
                3, 3, "Processing search results", doc_count
            )

            # Stream final response with tool results
            final_response_stream = self.tool_bound_chat_client.stream(messages)
            for chunk in final_response_stream:
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content

        else:
            # No tools called - stream direct response
            response_stream = self.chatClient.stream(messages)
            for chunk in response_stream:
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content

    def is_db_empty(self) -> bool:
        return self.vectorStore._collection.count() == 0

    def vector_search(self, query: str, k: int = 3) -> str:
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
            return self.vector_search(query, k)

        return vector_search_tool

    def _yield_progress_indicators(
        self, step: int, total_steps: int, action: str, doc_count=None
    ):
        """Yield progress indicators for tool execution (Options 1 & 2 only)"""
        # Option 1: Status emoji + text
        status_emoji = "🔍" if step < total_steps else "✅"
        yield f"{action}\n"

        # Option 2: Step counter
        yield f"Step {step}/{total_steps}: {action}\n"

        # Document count for final step
        if doc_count is not None and step == total_steps:
            plural = "" if doc_count == 1 else "s"
            yield f"Found {doc_count} relevant document{plural}. Generating response...\n"

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
    openaiAgent = Agent()

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
                st.session_state["answer"] = st.write_stream(openaiAgent.GenOpenAI(userPrompt))

        # TTS button (appears right after response)
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
