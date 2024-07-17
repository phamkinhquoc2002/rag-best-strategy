from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine, MultiStepQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform, StepDecomposeQueryTransform
from typing import Literal, Optional, Any

Mode = Literal['none', 'hyDE', 'multi']

class QueryEngine:
    """
    A class for handling different query engine modes.

    This class provides functionality for querying using different transformation modes:
    'none' (default retriever), 'hyDE' (Hypothetical Document Embeddings), or 'multi' (multi-step querying).

    Attributes:
        retriever: The retriever object used for querying.
        transform_mode (Mode): The mode of transformation to be applied to the query.
        node_processor: The reranker if needed.
        query_engine (RetrieverQueryEngine): The base query engine.
        hyde_query (TransformQueryEngine): The HyDE query engine (if applicable).
        multi_query (MultiStepQueryEngine): The multi-step query engine (if applicable).
    """

    def __init__(self, retriever: Any, transform_mode: Mode, llm: Any, node_processor:Optional[Any]):
        """
        Initialize the QueryEngine.

        Args:
            retriever: The retriever object to be used for querying.
            transform_mode (Mode): The mode of transformation to be applied to the query.
            llm: The language model to be used for transformations (required for 'hyDE' and 'multi' modes).
        """
        self.retriever = retriever
        self.transform_mode = transform_mode
        self.query_engine = RetrieverQueryEngine(self.retriever, 
                                                 node_postprocessors=[node_processor] 
                                                 if node_processor is not None else None)
        if self.transform_mode == 'hyDE' and llm is not None:
            hyde = HyDEQueryTransform(llm=llm)
            self.hyde_query = TransformQueryEngine(self.query_engine, hyde)
        elif self.transform_mode == 'multi' and llm is not None:
            step_decompose = StepDecomposeQueryTransform(llm=llm, verbose=True)
            self.multi_query = MultiStepQueryEngine(
                query_engine=self.query_engine,
                query_transform=step_decompose,
            )

    def query(self, query: str) -> None:
        """
        Execute a query based on the selected transformation mode.

        Args:
            query (str): The query string to be processed.

        Returns:
            None: This method prints the response or an error message.
        """
        if self.transform_mode == 'none':
            response = self.query_engine.query(query)
            return response
        elif self.transform_mode == 'hyDE':
            try:
                response = self.hyde_query.query(query)
                return response
            except AttributeError:
                print('You did not create the hyDE transform')
        elif self.transform_mode == 'multi':
            try:
                response = self.multi_query.query(query)
                return response
            except AttributeError:
                print('You did not create the multi-query transformation')