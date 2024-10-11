from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader,WebBaseLoader
from langchain.agents import AgentExecutor, create_structured_chat_agent

load_dotenv()

current_dir=os.path.dirname(os.path.abspath(__file__))
db_dir=os.path.join(current_dir, 'db')
persistent_directory=os.path.join(db_dir, 'chroma_database')
book=PyPDFLoader('attention.pdf')
book_docs=book.load()
web_new=["https://www.moneycontrol.com/","https://economictimes.indiatimes.com/"]
web_loader=WebBaseLoader(web_new)
web_documents=web_loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
book_split=text_splitter.split_documents(book_docs)
document_book=book_split[:20]

web_split=text_splitter.split_documents(web_documents)
web_docs=web_split[:10]


embedding_model=OllamaEmbeddings(model="llama3.2")
book_embedding=embedding_model.embed_documents(document_book)

web_embeddings=embedding_model.embed_documents(web_docs)

persistent_directory_1 = 'bookdb'
if not os.path.exists(persistent_directory_1):
    os.makedirs(persistent_directory_1)

# Directory for the second set of documents
persistent_directory_2 = 'webdb'
if not os.path.exists(persistent_directory_2):
    os.makedirs(persistent_directory_2)

db1 = Chroma.from_documents(
    book_embedding,
    persist_directory=persistent_directory_1
)
db1.persist()
#to save data

# Store second set of documents in another Chroma vector store
db2 = Chroma.from_documents(
    web_embeddings,
    persist_directory=persistent_directory_2
)
db2.persist()

class RetrieveTool1(BaseTool):
    name = "retrieve1"
    description = "Retrieves data from db1 based on a query using similarity search."
    
    def __init__(self):
        super().__init__()

    def _run(self, query: str):
        # Use the db1 retriever to get results
        retrieve1 = db1.as_retriever(
            search_type="similarity_search",
            search_kwargs={"k": 3},
        )
        return retrieve1.get_relevant_documents(query)


class RetrieveTool2(BaseTool):
    name = "retrieve2"
    description = "Retrieves data from db2 based on a query using similarity search."

    def __init__(self):
        super().__init__()

    def _run(self, query: str):
        retrieve2 = db2.as_retriever(
            search_type="similarity_search",
            search_kwargs={"k": 3},
        )
        return retrieve2.get_relevant_documents(query)


# Instantiate tools
tool1 = RetrieveTool1()
tool2 = RetrieveTool2()
tools = [tool1, tool2]

llm="x"
agent=create_structured_chat_agent(llm=llm,tools=tools)
# def combine_tool(query):
#     try:
#         data_retrieve1=tool1._run(query)
#         data_retrieve2=tool2._run(query)
#         combined_data=data_retrieve1+data_retrieve2
#         llm_input=" ".join(combined_data)
#         llm_response=agent.invoke({"input":llm_input})
#         return llm_response
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None



query="what is the latest news on ai?"
result=combine_tool(query)






