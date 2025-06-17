import os
from flask import Flask, request, render_template
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.together import TogetherLLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Setup Flask
app = Flask(__name__)

# Load API key
api_key = os.getenv("TOGETHER_API_KEY")

# Global query engine (we'll lazily initialize it)
query_engine = None

def initialize_index():
    global query_engine
    if query_engine is None:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        llm = TogetherLLM(model="meta-llama/Llama-3-8b-chat-hf", api_key=api_key)
        Settings.llm = llm
        Settings.embed_model = embed_model

        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

@app.route("/", methods=["GET", "POST"])
def chat():
    initialize_index()
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        response = query_engine.query(question)
        answer = str(response)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)
