from typing import Any
from utils import ChunkerStrategy, Faiss, data_loader, Retriever, fusion_retriever, QueryEngine, Reranker
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm(model:str, **kwargs) -> Any:
    model_name = kwargs.get('model_name')
    if model == "gemini":
        try:
            GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
            llm = Gemini(model=model_name)
        except KeyError:
            print("GOOGLE_API_KEY does not exist.")
    elif model == "openai":
        try:
            OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
            llm = OpenAI(model=model_name)
        except KeyError:
            print("OPENAI_API_KEY doesn't exist")
    elif model == "open-source":
        try: 
            HF_TOKEN = os.environ["HUGGING_FACE_TOKEN"]
            llm = HuggingFaceLLM(model_name=model_name)
        except KeyError:
            print("HUGGINGFACE TOKEN DOESN'T EXIST")
    elif model == "local":
        timeout = kwargs.get('timeout')
        llm = Ollama(model=model_name, request_timeout=timeout)
    return llm

def get_embedding(sentence_transformer: str) -> Any:
    embed_model = HuggingFaceEmbedding(model_name=sentence_transformer)
    return embed_model


   
if __name__ == '__main__':
    llm = get_llm(model = "gemini-pro")
    Settings.llm = llm
    Settings.embed_model = get_embedding(sentence_transformer="BAAI/bge-small-en-v1.5")
    print("Embed model and LLM loaded successfully")
    
    documents = data_loader(file_path='./documents')
    print("Documents loaded successfully")
    
    chunker = ChunkerStrategy(strategy='sentence')
    chunker = chunker.parser(chunk_size = 512, chunk_overlap=50)
    print("Chunker loaded successfully")
    
    index, doc = Faiss(documents=documents, dimension=384, transformation=[chunker])
    print("Index Loaded Successfully")
    
    rt1 = Retriever(vector=index, method='BM25')
    rt2 = Retriever(vector=index, method='vector')
    retriever1 = rt1.parser(docstore=doc)
    retriever2 = rt1.parser(docstore=doc)
    fusion = fusion_retriever(top_k=5, retrievers=[retriever1, retriever2], weights=[0.7, 0.3])
    print("Retriever Loaded Successfully")
    
    reranker = Reranker(strategy='custom').parser(top_n=3, model_name="cross-encoder/ms-marco-MiniLM-L-2-v2")
    query_engine = QueryEngine(retriever=fusion, transform_mode='none', llm=llm, node_processor=reranker)
    print("Query Engine Completion")
    
    query = 'The process of a GEN AI Cycle?'
    nodes = fusion.retrieve(query)
    print('\n--------------------------\n')
    for i, node in enumerate(nodes):
        print(f'{node.text}')
        print(f'{node.score}')
        print(f'\n----------The {i}th Node----------------\n')
    response = query_engine.query(query)
    print(response.get_formatted_sources(length=50))
    
    