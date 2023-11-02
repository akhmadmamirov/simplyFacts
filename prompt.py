from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

#Using existing Chroma db
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.run("What's an interesting fact about English language?")
print(result)

