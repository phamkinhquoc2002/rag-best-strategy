from typing import Literal, Any
from llama_index.core.postprocessor import SentenceTransformerRerank, LLMRerank, MetadataReplacementPostProcessor
from llama_index.postprocessor.cohere_rerank import CohereRerank

RerankerStrategy = Literal['llm-reranker', 'cohere', 'metadata', 'custom']

class Reranker:
    """
    A class to select and apply different reranking strategies.

    This class provides methods to rerank search results using different strategies:
    'custom', 'llm-reranker', 'metadata', or 'cohere'.

    Attributes:
        strategy (RerankerStrategy): The chosen reranking strategy.
    """

    def __init__(self, strategy: RerankerStrategy):
        """
        Initialize the Reranker with a specific strategy.

        Args:
            strategy (RerankerStrategy): The reranking strategy to use.
        """
        self.strategy = strategy

    def parser(self, **kwargs: Any) -> Any:
        """
        Parse the chosen strategy to the appropriate reranking method.

        Args:
            **kwargs: Arbitrary keyword arguments for the specific reranker.

        Returns:
            Any: The appropriate reranker based on the chosen strategy.

        Raises:
            ValueError: If an invalid strategy is provided.
        """
        if self.strategy == 'custom':
            return self.sentence_transformer(**kwargs)
        elif self.strategy == 'llm-reranker':
            return self.llm_ranker(**kwargs)
        elif self.strategy == 'metadata':
            return self.metadata_replacement(**kwargs)
        elif self.strategy == 'cohere':
            return self.cohere_reranker(**kwargs)
        else:
            raise ValueError(f"Invalid reranking strategy: {self.strategy}")

    def sentence_transformer(self, **kwargs: Any) -> SentenceTransformerRerank:
        """
        Create a SentenceTransformerRerank object.

        Args:
            **kwargs: Arbitrary keyword arguments.
                model_name (str): The name of the sentence transformer model.
                top_n (int): The number of top results to return.

        Returns:
            SentenceTransformerRerank: A configured SentenceTransformerRerank object.
        """
        model_name = kwargs.get('model_name')
        top_n = kwargs.get('top_n')
        return SentenceTransformerRerank(top_n=top_n, model=model_name)

    def llm_ranker(self, **kwargs: Any) -> LLMRerank:
        """
        Create an LLMRerank object.

        Args:
            **kwargs: Arbitrary keyword arguments.
                top_n (int): The number of top results to return.

        Returns:
            LLMRerank: A configured LLMRerank object.
        """
        top_n = kwargs.get('top_n')
        return LLMRerank(top_n=top_n)

    def metadata_replacement(self, **kwargs: Any) -> MetadataReplacementPostProcessor:
        """
        Create a MetadataReplacementPostProcessor object.

        Args:
            **kwargs: Arbitrary keyword arguments (unused in this method).

        Returns:
            MetadataReplacementPostProcessor: A configured MetadataReplacementPostProcessor object.
        """
        return MetadataReplacementPostProcessor(target_metadata_key="window")

    def cohere_reranker(self, **kwargs: Any) -> CohereRerank:
        """
        Create a CohereRerank object.

        Args:
            **kwargs: Arbitrary keyword arguments.
                api_key (str): The API key for Cohere.
                top_n (int): The number of top results to return.

        Returns:
            CohereRerank: A configured CohereRerank object.
        """
        api_key = kwargs.get('api_key')
        top_n = kwargs.get('top_n')
        return CohereRerank(api_key=api_key, top_n=top_n)