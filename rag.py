from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.pipeline import Pipeline
from retriever import retriever

template = """
     Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{query}}
    \nAnswer:
"""

prompt_builder = PromptBuilder(template=template)
generator = OllamaGenerator(
    model = "mistral",
    url = "http://localhost:11434/api/generate",
    generation_kwargs = {"num_predict": 100, "temperature":0.9})


# basic pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever )
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", generator)
rag_pipeline.connect("retriever", "prompt_builder.documents")

rag_pipeline.connect("prompt_builder", "llm")


if __name__ == "__main__":

    question = "What does Rhodes statue look like?"

    response = rag_pipeline.run(
        {
            "retriever":{"query": question},
            "prompt_builder": {"query": question},

        }

    )
    print(response["llm"]["replies"][0])





