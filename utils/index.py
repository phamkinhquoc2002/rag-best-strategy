import os
from typing import Literal, List, Tuple, Optional, Any
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    SimpleKeywordTableIndex,
    Document
)
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

Strategy = Literal['sentence', 'semantic', 'window', 'hierarchical']

def data_loader(file_path: str) -> List[Document]:
    """
    Load documents from a specified directory.

    Args:
        file_path (str): The path to the directory containing the documents.

    Returns:
        List[Document]: A list of loaded documents.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No directory found at {file_path}")
    
    reader = SimpleDirectoryReader(input_dir=file_path, filename_as_id=True)
    documents = reader.load_data()
    return documents

def Faiss(documents: List[Document], dimension: int, transformation: Optional[List] = None) -> Tuple[VectorStoreIndex, Any]:
    """
    Create or load a Faiss vector database index.

    This function either loads an existing Faiss index from storage or creates a new one
    if it doesn't exist.

    Args:
        documents (List[Document]): The documents to be indexed.
        dimension (int): The dimension of the vector space.
        transformation (Optional[List]): A list of transformations to apply to the documents.
    """
    faiss_index = faiss.IndexFlatL2(dimension)
    
    if os.path.exists("./storage"):
        vector_store = FaissVectorStore.from_persist_dir("./storage")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir="./storage"
        )
        index = load_index_from_storage(storage_context=storage_context)
    else:
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, 
            transformations=transformation,
            storage_context=storage_context
        )
        index.storage_context.persist()
    
    return index, storage_context.docstore