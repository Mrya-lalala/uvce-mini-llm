import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from llama_index.core import Settings
from flask import Flask, request, render_template
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.together import TogetherLLM
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

# Setup embedding + LLM
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = TogetherLLM(model="meta-llama/Llama-3-8b-chat-hf", api_key=api_key)
Settings.llm = llm
Settings.embed_model = embed_model

# Load docs + index
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Setup Flask
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def chat():
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        response = query_engine.query(question)
        answer = str(response)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))  # Render uses PORT
    app.run(debug=True, host="0.0.0.0", port=port)