import gradio as gr
from rag import rag_pipeline

def ask_question(question):
     response = rag_pipeline.run(
        {
            "retriever":{"query": question},
            "prompt_builder": {"query": question},

        }
    )
     return response["llm"]["replies"][0]

if __name__ == "__main__":
    app = gr.Interface(fn=ask_question,
                       inputs=gr.components.Textbox(lines=2,
                                                placeholder = "Enter your question here",),
                                                outputs="text")
    app.launch()