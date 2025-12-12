# utility imports
from typing import Callable
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
from tools.websearching import perpSearch
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
            base_url="http://172.16.5.234:3000", model="qwen3-vl:4b"
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

        # Create the search tools (binding happens dynamically in GenOllama)
        self.vectorToolInstance = self.createTool(
            self.vector_search_tool, "vector store", "vectorToolInstance"
        )
        self.webToolInstance = self.createTool(perpSearch, "web", "webToolInstance")

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

    def GenOllama(self, question: str, enable_web_search: bool = False):
        """Tool-calling agent that can dynamically search documents."""
        # Dynamic system prompt and tool binding based on web search setting
        if enable_web_search:
            tools = [self.vectorToolInstance, self.webToolInstance]
            system_prompt = """You are an academic research companion with access to multiple information sources.

Available tools:
- vectorToolInstance: Search through uploaded PDF documents for specific information
- webToolInstance: Search the web for current information not available in local documents

Guidelines:
- Use vectorToolInstance when answering questions about uploaded documents
- Use webToolInstance when you need current web information or the question requires up-to-date data
- For general knowledge questions: Answer directly from your training
- Always cite sources when using search results
- If no relevant information exists, clearly state this

Be concise, academic, and evidence-based in your responses."""
        else:
            tools = [self.vectorToolInstance]
            system_prompt = """You are an academic research companion specializing in analyzing PDF documents.

Available tools:
- vectorToolInstance: Search through uploaded PDF documents for specific information

Guidelines:
- Use vectorToolInstance when answering questions about uploaded documents
- For general knowledge questions: Answer directly from your training
- Always cite sources when using search results
- If no relevant information exists, clearly state this

Be concise, academic, and evidence-based in your responses."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]

        # Create dynamic chat client with appropriate tools
        chat_client = self.chatClient.bind_tools(tools)

        # Get initial LLM response
        response = chat_client.invoke(messages)

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
                elif tool_call["name"] == "webToolInstance":
                    tool_result = self.webToolInstance.invoke(tool_call["args"])
                    messages.append(
                        ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                    )

            # Get final response with tool results
            final_response = chat_client.invoke(messages)
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

    def createTool(self, importedFunction: Callable, desciptVal: str, toolName: str):
        """Create the vector search tool for binding to the LLM."""

        @tool
        def dynamicTool(query: str, k: int = 3) -> str:
            """Search for relevant information based on the query."""
            return importedFunction(query, k)

        # Set unique name and description to avoid tool collisions
        dynamicTool.name = toolName
        dynamicTool.description = (
            f"Search the {desciptVal} for relevant document chunks based on the query."
        )
        return dynamicTool


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

    # Checking if chroma db is empty
    if ollamaAgent.vectorStore._collection.count() == 0:
        st.info("The vector store is empty, upload something", icon="ℹ️")
    # Streamlit UI
    uploadedPdf = st.file_uploader(
        "Upload a pdf please", type="pdf"
    )  # uploading the pdf

    # Web search toggle
    enable_web_search = st.checkbox(
        "🌐 Enable Web Search",
        value=False,  # Default OFF for speed
        help="Allow AI to search the web for additional information. May slow down responses.",
    )

    userPrompt = st.text_area("Prompting")
    if userPrompt:
        answerBtn = st.button("Send prompt")
        if answerBtn:
            with st.spinner("Processing llm..."):
                answer = ollamaAgent.GenOllama(
                    userPrompt, enable_web_search=enable_web_search
                ).content
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
