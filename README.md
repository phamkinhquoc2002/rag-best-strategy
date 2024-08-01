# Experiment: Testing Different RAG Strategies
Inspired by the paper: "Searching for Best Practices in Retrieval-Augmented Generation" by Wang et al. This repository is dedicated to search for the best RAG strategy on a tight budget.

## How to run it locally
You can use the repository on your local laptop to play around with different RAG pipelines.

### Installation and Setup
1. Create a virtual environment
```
python -m venv venv
./venv/Scripts/Activate
```
2. Install the requirements:
 ```
pip install -r requirements.txt
```
3. Specify the different API keys in .env:
 ```
GOOGLE_API_KEY: for google models
OPENAI_API_KEY: for openai models
HF_TOKEN: for open-sourced models
```
### Choose the data and RAG strategy
4. Put the documents that you want to query into a documents folder
5. Play around with the code in test.py:
   Example using gemini-pro and BAAI/bge-small-en-v1.5:
 ```
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
```

## Evaluation Strategy
Each RAG strategy was evaluated by Trulens Evaluation Benchmarks: RAG Triad. The RAG triad is made up of 3 evaluations: context relevance, groundedness and answer relevance. Satisfactory evaluations on each provides us confidence that our LLM app is free from hallucination.

![rag_triad](rag_evaluation.jpg)

## Searching for Best Practices in Retrieval-Augmented Generation
<div align="center" style="margin-top: 20px; margin-bottom: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
  <img src="quick_read.jpg" alt="Paper Description">
</div>

```
@inproceedings{Wang2024SearchingFB,
  title={Searching for Best Practices in Retrieval-Augmented Generation},
  author={Xiaohua Wang and Zhenghua Wang and Xuan Gao and Feiran Zhang and Yixin Wu and Zhibo Xu and Tianyuan Shi and Zhengyuan Wang and Shizheng Li and Qi Qian and Ruicheng Yin and Changze Lv and Xiaoqing Zheng and Xuanjing Huang},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:270870251}
}
```
