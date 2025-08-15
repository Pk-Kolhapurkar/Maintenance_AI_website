import os
import gradio as gr
import requests
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain.embeddings import HuggingFaceEmbeddings

# ----------- 1. Custom LLM to call your LitServe endpoint -----------
class LitServeLLM(LLM):
    endpoint_url: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {"prompt": prompt}
        response = requests.post(self.endpoint_url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()
        else:
            raise ValueError(f"Request failed: {response.status_code} {response.text}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"endpoint_url": self.endpoint_url}

    @property
    def _llm_type(self) -> str:
        return "litserve_llm"


# ----------- 2. Connect to Pinecone -----------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("rag-granite-index")

# ----------- 3. Load embedding model -----------
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------- 4. Function to get top context from Pinecone -----------
def get_retrieved_context(query: str, top_k=3):
    query_embedding = embeddings_model.embed_query(query)
    results = index.query(
        namespace="rag-ns",
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    context_parts = [match['metadata']['text'] for match in results['matches']]
    return "\n".join(context_parts)

# ----------- 5. Create LLMChain with your model -----------
model = LitServeLLM(
    endpoint_url="https://8001-01k2h9d9mervcmgfn66ybkpwvq.cloudspaces.litng.ai/predict"
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a smart assistant. Based on the provided context, answer the question in 1–2 lines only.
If the context has more details, summarize it concisely.
Context:
{context}
Question: {question}
Answer:
"""
)

llm_chain = LLMChain(llm=model, prompt=prompt)

# ----------- 6. Main RAG Function -----------
def rag_pipeline(question):
    try:
        retrieved_context = get_retrieved_context(question)
        response = llm_chain.invoke({
            "context": retrieved_context,
            "question": question
        })["text"].strip()

        # Only keep what's after "Answer:"
        if "Answer:" in response:
            response = response.split("Answer:", 1)[-1].strip()

        return response
    except Exception as e:
        return f"Error: {str(e)}"


# ----------- 7. Gradio UI -----------
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 RAG Chatbot (Pinecone + LitServe)")
    question_input = gr.Textbox(label="Ask your question here")
    answer_output = gr.Textbox(label="Answer")
    ask_button = gr.Button("Get Answer")
    ask_button.click(rag_pipeline, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    demo.launch()
