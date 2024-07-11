from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser, SemanticSplitterNodeParser, HierarchicalNodeParser 
from typing import Literal, Any

node_parser_strategy = Literal['sentence', 'window', 'semantic', 'hiearchical']

class ChunkerStrategy():
    
    def __init__(self, strategy: node_parser_strategy):
        """
        Select the best strategy for chunking process.
        """
        self.strategy = strategy
        
    def parser(self, *args: Any, **kwargs:Any):
        """
        
        Parse the chosen strategy to relative function.
        """
        if self.strategy == 'sentence':
            return self._sentence_splitter(*args, **kwargs)
        elif self.strategy == 'window':
            return self._window_parser(*args, **kwargs)
        elif self.strategy == 'semantic':
            return self._semantic_chunker(*args, **kwargs)    
        elif self.strategy == 'hiearchical':
            return self._hiearchical_chunker(*args, **kwargs)
        
    def _sentence_splitter(self, *args: Any, **kwargs: Any) -> Any:
        chunk_size = kwargs.get('chunk_size')
        chunk_overlap = kwargs.get('chunk_overlap')
        
        if chunk_size is None or chunk_overlap is None:
            raise ValueError("chunk_size and chunk_overlap must be provided for this strategy")
        return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  
    
    def _window_parser(self, *args: Any, **kwargs: Any) -> Any:
        window_size = kwargs.get('window_size')
        return SentenceWindowNodeParser(window_size=window_size, window_metadata_key="window", original_text_metadata_key="original_sentence")
    
    def _semantic_chunker(self, *args: Any, **kwargs: Any) -> Any:
        buffer_size = kwargs.get('buffer_size')
        threshold = kwargs.get('threshold')
        embed_model = kwargs.get('embed_model')
        return SemanticSplitterNodeParser(buffer_size=buffer_size, breakpoint_percentile_threshold=threshold, embed_model=embed_model)
    
    def _hiearchical_chunker(self, *args: Any, **kwargs: list) -> Any:
        chunk_sizes=kwargs.get('chunk_sizes')
        return HierarchicalNodeParser(chunk_sizes=chunk_sizes)