import argparse
from utils import ChunkerStrategy, Faiss, data_loader, Retriever, fusion_retriever, QueryEngine, Reranker
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')

from llama_index.llms.gemini import Gemini
llm = Gemini(model="models/gemini-pro")
Settings.embed_model = embed_model
Settings.llm = llm

def arg_parse():
    parser=argparse.ArgumentParser(description='Testing')
    parser.add_argument('--embedding', '--embed', type='str', help='Embedding Model Selection')
    args=parser.parse_args()
    return args
       
if __name__ == '__main__':
    documents = data_loader(file_path='./documents')
    print("documents loaded")
    chunker = ChunkerStrategy(strategy='sentence')
    chunker = chunker.parser(chunk_size = 512, chunk_overlap=50)
    index, doc = Faiss(documents=documents, dimension=384, transformation=[chunker])
    rt1 = Retriever(vector=index, method='BM25')
    rt2 = Retriever(vector=index, method='vector')
    retriever1 = rt1.parser(docstore=doc)
    retriever2 = rt1.parser(docstore=doc)
    fusion = fusion_retriever(top_k=5, retrievers=[retriever1, retriever2], weights=[0.7, 0.3])
    reranker = Reranker(strategy='custom').parser(top_n=3, model_name="cross-encoder/ms-marco-MiniLM-L-2-v2")
    query_engine = QueryEngine(retriever=fusion, transform_mode='none', llm=llm, node_processor=reranker)
    query = 'The process of a GEN AI Cycle?'
    nodes = fusion.retrieve(query)
    print('\n--------------------------\n')
    for i, node in enumerate(nodes):
        print(f'{node.text}')
        print(f'{node.score}')
        print(f'\n----------The {i}th Node----------------\n')
    response = query_engine.query(query)
    print(response.get_formatted_sources(length=50))
    
    