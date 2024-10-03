
import os
from typing import Tuple, List, Any
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
# from langchain.storage import LocalFileStore
from chainlit.types import AskFileResponse
# from langchain.embeddings import CacheBackedEmbeddings
# from qdrant_client.http.models import Distance, VectorParams
# from .agents.context_gathering import context_gathering
# import sys
# sys.path.append('./assignments/week_02/proposal_generation_agent/')
# from agents.context_gathering import context_gathering
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.context_gathering import context_gathering
sys.path.append('./agents/workflowLlamaIndex/')
# from workflowLlamaIndex.CRAGLlamaIndexWorkflow import run_workflow
from workflowLlamaIndex.CRAGLlamaIndexWorkflow import CorrectiveRAGLlamIndexWorkflow
# from qdrant_client import QdrantClient
import chainlit as cl
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.config import RunnableConfig
from langchain.schema.runnable import Runnable
from dotenv import load_dotenv
import uuid
from uuid import uuid4



# Load environment variables from .env file
load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

#------------------------------
# Prompt Template
#-----------------------------

rag_system_prompt_template = """\
You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existance of context. 
"""

rag_user_prompt_template = """\
Question: 
{question}
context:
{context}
"""


chat_prompt: List[Tuple [str, ChatPromptTemplate]] = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt_template),
     ("user", rag_user_prompt_template)
])

#---------------------------
# llm model initialization
#--------------------------

chat_model: ChatOpenAI=ChatOpenAI(model="gpt-4o-mini")

#-------------------------
# process file
#------------------------
def process_file(file: AskFileResponse) -> List[Document]:
    # Loader: PyMuPDFLoader=
    loader: PyMuPDFLoader=PyMuPDFLoader(file.path)
    documents: List[Any]=loader.load()
    docs: List[Document]=text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"]=f"source_{i}"
    
    return docs



#------------------------------------------------------
# chainlit
#-----------------------------------------------------
import chainlit as cl


@cl.on_chat_start
async def start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    
    msg = cl.Message(
        content=f"Processing `{file.name}`...",
    )
    await msg.send()

    #-------------------------------------------
    # Load file to metadata with reference page
    #------------------------------------------
    # docs: List[Document]=process_file(file)

    #----------------------------------------------------------
    # gather into vector database
    #-----------------------------------------------------------
    # Create a Qdrant vector store with cache backed embeddings
    #-----------------------------------------------------------
    #True: Qdrant
    #False: Faiss
    ###################################################################################
    #-----------------------------------------------------------
    # retriever=context_gathering(docs,
    #                             use_qdrant=True)#,
                                #collection_name=f"pdf_to_parse_{uuid.uuid4()}")
                                # Create a Qdrant vector store with cache backed embeddings
    #-----------------------------------------------------------
    # Create a chain that uses the QDrant/faiss vector store
    # Parallelization: LCEL runnables are parallelized by default, allowing for efficient
    # execution of multiple steps in the chain simultaneously, improving overall performance.
    #----------------------------------------------------------
    # retrieval_augmented_qa_chain: Runnable = (
    #     {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    #     | RunnablePassthrough.assign(context=itemgetter("context"))
    #     | chat_prompt | chat_model
    # )
    #################################################################################
    CRAGAgent=CorrectiveRAGLlamIndexWorkflow()

    from llama_index.core import Document as LlamaIndexDocument
    docs: List[Document] = process_file(file)
    
    # # print(f"Document attributes: {dir(docs[0])}")
    print(f"Document representation:\n {repr(docs[0])}")

    def convert_to_llama_index_docs(docs):
        return [LlamaIndexDocument(text=doc.page_content, metadata=doc.metadata) for doc in docs]

    # In your start function
    # docs = process_file(file)
    llama_index_docs = convert_to_llama_index_docs(docs)

    # index = await CRAGAgent.run(documents=llama_index_docs)

    

    index = await CRAGAgent.run(documents=llama_index_docs)
    print("Workflow run completed.\n")
    print(index)

    cl.user_session.set("LlamaLang", CRAGAgent)
    cl.user_session.set("index", index)

    # Let the user know that the system is ready
    msg.content = f"Processing {file.name} done! You can ask questions now!"
    await msg.update()

    # except Exception as e:
    #     print(f"Error during workflow execution: {e}")
    #     raise
    
    # # Save the list in the session to store the message history
    # cl.user_session.set("inputs", {"messages": []})

    # # Create a thread id and pass it as configuration 
    # # to be able to use Langgraph's MemorySaver
    conversation_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": conversation_id}}
    cl.user_session.set("config", config)



#----------------------------------------------------------------------------------------------------
# Decorator: This Chainlit decorator is used to rename the authors of messages in the chat interface
##----------------------------------------------------------------------------------------------------
@cl.author_rename
def rename(orig_author: str):
    rename_dict={"ChatOpenAI": "The Genrator...", "VectorStroeRetriever": "The Reriever..."}
    return rename_dict.get(orig_author,orig_author)

@cl.on_message
async def main(message: cl.Message):  
#    runnable=cl.user_session.get("graph")
#    msg=cl.Message(content="")
#------------------------------------------------------
# Async method: Using astream allows for asynchronous streaming of the response,
# improving responsiveness and user experience by showing partial results as they become available.
#    async for chunk in runnable.astream(
       
#         {"question": message.content},
#         config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
#     ):
#         await msg.stream_token(chunk.content)
#    await msg.send()
#####################################################
    AgentCRAG = cl.user_session.get("LlamaLang")
    index = cl.user_session.get("index")
    
    msg=cl.Message(content="")


    # response =await AgentCRAG.run(
    #     query_str=message.content,
    #     index=index,
    #     tavily_ai_apikey=os.getenv("TAVILY_API_KEY"), config="config",
    # )
    # await cl.Message(content=str(response)).send()
    response= await AgentCRAG.run(
    
        query_str= message.content,
        index=index,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    )
    # Process the response
    # await msg.stream_token(response)
    await cl.Message(content=str(response)).send()



##################################################################








