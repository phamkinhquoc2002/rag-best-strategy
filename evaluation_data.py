from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

def testset_generator(documents, generator: str, critic: str, embed_model: str, **kwargs):
    generator = TestsetGenerator.from_llama_index(
        generator_llm=generator,
        critic_llm=critic,
        embeddings=embed_model
    )
    
    simple, reasoning, multi_context = kwargs.get('test_set_parameters')
    
    testset = generator.generate_with_llamaindex_docs(
        documents,
        test_size=5,
        distributions={simple: simple, reasoning: reasoning, multi_context: multi_context},
        )
    
    ds = testset.to_dataset()
    ds_dict = ds.to_dict()
    return ds_dict