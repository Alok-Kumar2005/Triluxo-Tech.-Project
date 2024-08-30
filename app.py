from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key = os.getenv("GENAI_API_KEY"))


url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')

# creating chunk of text and then store in faiss database
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")


# creating flask app
app = Flask(__name__)

vector_store = FAISS.load_local("faiss_index", embeddings)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    response = vector_store.similarity_search(user_input)
    return jsonify({'response': response[0].page_content})

if __name__ == "__main__":
    app.run(debug=True)