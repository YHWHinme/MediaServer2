# utility imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from uuid import uuid4
#RAG
from langchain_chroma import Chroma
# Ai MOdels
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings


# Models
embeddings = OllamaEmbeddings(
    base_url="http://172.16.5.60:3000",
    model="embeddinggemma:300m"
)

baseFilepath = "./Data/deps/pdfs"
# File processing

vector_store = Chroma(
    collection_name="pdf_collection",
    embedding_function=embeddings,
    persist_directory="./Data/Chroma",  # Where to save data locally, remove if not necessary
)

def loadPdf(pdf: PyPDFLoader, storeClient):
    # Initing values
    docs = pdf.load() # Unpacking the pdf
    result_docs = []
    # Creating reliable documents
    for page in docs:
        pageLabel = page.metadata["page_label"]

        document = Document(
            page_content= "Page: " + pageLabel + "\n\n" + page.page_content,
            metadata={"page_label": page.metadata["page_label"], "source": page.metadata["source"]}
        )

        result_docs.append(document)

    # Creating id for vector store
    uuids = [str(uuid4()) for _ in range(len(result_docs))]

    storeClient.add_documents(documents=result_docs, ids=uuids)

