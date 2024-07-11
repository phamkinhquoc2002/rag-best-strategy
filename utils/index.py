from typing import Literal
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import os

strategy = Literal['sentence', 'semantic', 'window', 'hiearchical']

def data_loader(file_path):
    if os.path.exists(file_path) is None:
        return "No files were found"
    reader = SimpleDirectoryReader(input_dir=file_path, filename_as_id=True)
    documents = reader.load_data()
    return documents
    
def Faiss(documents, dimension, transformation):
    """
    Vector Database Call: Faiss
    """
    faiss_index = faiss.IndexFlatL2(dimension)
    if os.path.exists("./storage") is not None:
        vector_store = FaissVectorStore.from_persist_dir("./storage")
        storage_context = StorageContext.from_defaults(vector_store=vector_store, 
                                                       persist_dir="./storage")
        index = load_index_from_storage(storage_context=storage_context)
        return index
    else:
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, transformations=transformation,storage_context=storage_context)
        index.storage_context.persist()
        return index

    
    