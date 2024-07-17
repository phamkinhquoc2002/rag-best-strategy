from llama_index.core.node_parser import (
    SentenceSplitter,
    SentenceWindowNodeParser,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser
)
from typing import Literal, Any, Union

NodeParserStrategy = Literal['sentence', 'window', 'semantic']

class ChunkerStrategy:
    """
    A class to select and apply different text chunking strategies.

    This class provides methods to parse text using different chunking strategies:
    'sentence', 'window', or 'semantic'.

    Attributes:
        strategy (NodeParserStrategy): The chosen chunking strategy.
    """

    def __init__(self, strategy: NodeParserStrategy):
        """
        Initialize the ChunkerStrategy with a specific strategy.

        Args:
            strategy (NodeParserStrategy): The chunking strategy to use.
        """
        self.strategy = strategy

    def parser(self, *args: Any, **kwargs: Any) -> Union[SentenceSplitter, SentenceWindowNodeParser, SemanticSplitterNodeParser]:
        """
        Parse the chosen strategy to the relative function.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Union[SentenceSplitter, SentenceWindowNodeParser, SemanticSplitterNodeParser]: 
            The appropriate parser based on the chosen strategy.

        Raises:
            ValueError: If an invalid strategy is provided.
        """
        if self.strategy == 'sentence':
            return self._sentence_splitter(*args, **kwargs)
        elif self.strategy == 'window':
            return self._window_parser(*args, **kwargs)
        elif self.strategy == 'semantic':
            return self._semantic_chunker(*args, **kwargs)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

    def _sentence_splitter(self, *args: Any, **kwargs: Any) -> SentenceSplitter:
        """
        Create a SentenceSplitter with the provided parameters.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
                chunk_size (int): The size of each chunk.
                chunk_overlap (int): The overlap between chunks.

        Returns:
            SentenceSplitter: A configured SentenceSplitter object.

        Raises:
            ValueError: If chunk_size or chunk_overlap is not provided.
        """
        chunk_size = kwargs.get('chunk_size')
        chunk_overlap = kwargs.get('chunk_overlap')

        if chunk_size is None or chunk_overlap is None:
            raise ValueError("chunk_size and chunk_overlap must be provided for this strategy")
        return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _window_parser(self, *args: Any, **kwargs: Any) -> SentenceWindowNodeParser:
        """
        Create a SentenceWindowNodeParser with the provided parameters.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
                window_size (int): The size of the window.

        Returns:
            SentenceWindowNodeParser: A configured SentenceWindowNodeParser object.
        """
        window_size = kwargs.get('window_size')
        return SentenceWindowNodeParser(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_sentence"
        )

    def _semantic_chunker(self, *args: Any, **kwargs: Any) -> SemanticSplitterNodeParser:
        """
        Create a SemanticSplitterNodeParser with the provided parameters.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
                buffer_size (int): The size of the buffer.
                threshold (float): The breakpoint percentile threshold.
                embed_model: The embedding model to use.

        Returns:
            SemanticSplitterNodeParser: A configured SemanticSplitterNodeParser object.
        """
        buffer_size = kwargs.get('buffer_size')
        threshold = kwargs.get('threshold')
        embed_model = kwargs.get('embed_model')
        return SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=threshold,
            embed_model=embed_model
        )