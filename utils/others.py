from llama_index.core.retrievers import (
    QueryFusionRetriever
)

from typing import List, Any

def fusion_retriever(top_k: int, retrievers: List[Any], weights: List[float]) -> QueryFusionRetriever:
        """
        Create a QueryFusionRetriever with the provided parameters.

        Args:
            top_k (int): The number of top results to retrieve.
            retrievers (List[Any]): A list of retrievers to fuse.
            weights (List[float]): A list of weights for each retriever.

        Returns:
            QueryFusionRetriever: A configured QueryFusionRetriever object.
        """
        return QueryFusionRetriever(
            retrievers=retrievers,
            retriever_weights=weights,
            similarity_top_k=top_k,
            num_queries=1,
            mode='relative_score'
        )