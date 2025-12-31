# utility imports
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from uuid import uuid4
from pypdf import PdfReader

# Remove duplicate model setup - models will be passed from Agent class


def loadPdf(pdf: PyPDFLoader, storeClient):
    """Enhanced PDF processing with metadata extraction and organized storage"""
    temp_file_path = None

    try:
        # Extract document metadata using pypdf
        reader = PdfReader(pdf.file_path)
        doc_metadata = reader.metadata

        # Extract key fields with fallbacks
        title = getattr(doc_metadata, "title", None) or "Unknown_Title"
        author = getattr(doc_metadata, "author", None) or "Unknown_Author"
        subject = getattr(doc_metadata, "subject", None)
        creation_date = getattr(doc_metadata, "creation_date", None)

        # Sanitize for filesystem
        safe_title = title.replace("/", "_").replace("\\", "_").replace(" ", "_")[:100]
        safe_author = author.replace("/", "_").replace("\\", "_").replace(" ", "_")[:50]

        # Create simple filename for direct placement in pdfs folder
        safe_filename = f"{safe_author}_{safe_title}.pdf"
        permanent_path = Path("./Data/deps/pdfs") / safe_filename

        # Only copy if it doesn't exist (avoid duplicates)
        if not permanent_path.exists():
            import shutil

            shutil.copy2(pdf.file_path, permanent_path)

        # Check if this is a temp file that needs cleanup
        if hasattr(pdf, "file_path") and "Data/deps/temp" in pdf.file_path:
            temp_file_path = pdf.file_path

        # Load pages with enhanced metadata
        docs = pdf.load()
        result_docs = []

        for page in docs:
            enhanced_metadata = {
                "page_label": page.metadata["page_label"],
                "source": str(permanent_path),  # Use permanent direct path
                "document_title": title,
                "author": author,
                "subject": subject,
                "creation_date": str(creation_date) if creation_date else None,
                "folder_path": str(Path("./Data/deps/pdfs")),  # Direct pdfs folder
                "embedding_model": "text-embedding-ada-002",  # Track which model was used
            }

            document = Document(
                page_content=f"Title: {title}\nAuthor: {author}\nPage: {page.metadata['page_label']}\n\n{page.page_content}",
                metadata=enhanced_metadata,
            )
            result_docs.append(document)

        # Add to vector store using the passed OpenAI embedding function
        uuids = [str(uuid4()) for _ in range(len(result_docs))]
        storeClient.add_documents(documents=result_docs, ids=uuids)

    finally:
        # Clean up temp file after processing
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                print(f"Warning: Failed to cleanup temp file {temp_file_path}")
