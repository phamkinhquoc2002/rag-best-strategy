from llama_index.core.retrievers import (
    QueryFusionRetriever,
    AutoMergingRetriever
)
from llama_index.retrievers.bm25 import BM25Retriever
from typing import Literal, Any, List
import Stemmer

RetrievalMethod = Literal['vector', 'BM25', 'automerge']

class Retriever:
    """
    A class to select and apply different retrieval methods.

    This class provides methods to retrieve information using different strategies:
    'vector', 'BM25', or 'automerge'.

    Attributes:
        vectorIndex: The vector index used for retrieval.
        retrieval_method (RetrievalMethod): The chosen retrieval method.
    """

    def __init__(self, vector: Any, method: RetrievalMethod):
        """
        Initialize the Retriever with a vector index and a retrieval method.

        Args:
            vector: The vector index to be used for retrieval.
            method (RetrievalMethod): The retrieval method to use.
        """
        self.vectorIndex = vector
        self.retrieval_method = method

    def parser(self, **kwargs: Any) -> Any:
        """
        Parse the chosen strategy to retrieve information.

        Args:
            **kwargs: Arbitrary keyword arguments for the specific retriever.

        Returns:
            Any: The appropriate retriever based on the chosen method.

        Raises:
            ValueError: If an invalid retrieval method is provided.
        """
        if self.retrieval_method == 'vector':
            return self.vectorIndex.as_retriever(**kwargs)
        elif self.retrieval_method == 'BM25':
            return self._BM25_retriever(**kwargs)
        elif self.retrieval_method == 'automerge':
            return self._auto_merge_retriever(**kwargs)
        else:
            raise ValueError(f"Invalid retrieval method: {self.retrieval_method}")

    def _BM25_retriever(self, **kwargs: Any) -> BM25Retriever:
        """
        Create a BM25Retriever with the provided parameters.

        Args:
            **kwargs: Arbitrary keyword arguments.
                docstore: The document store to use.

        Returns:
            BM25Retriever: A configured BM25Retriever object.
        """
        docstore = kwargs.get('docstore')
        return BM25Retriever.from_defaults(
            docstore=docstore,
            similarity_top_k=2,
            stemmer=Stemmer.Stemmer("english"),
            language="english"
        )

    def _auto_merge_retriever(self, **kwargs: Any) -> AutoMergingRetriever:
        """
        Create an AutoMergingRetriever with the provided parameters.

        Args:
            **kwargs: Arbitrary keyword arguments.
                top_k (int): The number of top results to retrieve.
                storage_context: The storage context to use.

        Returns:
            AutoMergingRetriever: A configured AutoMergingRetriever object.
        """
        top_k = kwargs.get('top_k')
        storage_context = kwargs.get('storage_context')
        base_retriever = self.vectorIndex.as_retriever(similarity_top_k=top_k)
        return AutoMergingRetriever(base_retriever, storage_context, verbose=True)