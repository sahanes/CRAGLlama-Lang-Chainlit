import getpass
import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()
# os.environ["LANGCHAIN_PROJECT"] = "LlamaIndexworkflow"

VERSION = '1.1_rc3'
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT") + f" - v. {VERSION}"

from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool




import asyncio
import importlib
import sys
sys.path.append('./agents/workflowLlamaIndex/')
from workflowLlamaIndex.CRAGLlamaIndexWorkflow import run_workflow
from workflowLlamaIndex.CRAGLlamaIndexWorkflow import CorrectiveRAGLlamIndexWorkflow

from langchain_community.llms import OpenAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage


from chainlit.logger import logger
from realtime import RealtimeClient
from realtime.tools import tools

@tool
async def Llamarag(question: str) -> str:
    """Retrieve information ONLY from the uploaded PDF document."""
    # response = "Hallucinations are generally considered more harmful when they are associated with psychiatric illness or neurological conditions, as they can be symptoms of underlying health issues. In these cases, hallucinations can lead to distress, confusion, and impaired functioning. It is important to address hallucinations that are part of a psychiatric or neurological disorder promptly to prevent potential negative consequences."
    response= await run_workflow(question)
    # Ensure the response is wrapped in the framework's expected message format
    return str(response)#{'messages' :[AIMessage(content=response)]}#FunctionMessage(content=str(response), name="Llamarag")
    # return FunctionMessage(content=str(response), last_message.additional_kwargs["function_call"]["name"]

from langchain_core.utils.function_calling import convert_to_openai_function
# from langgraph.prebuilt import ToolNode
# from langgraph.graph import StateGraph, MessagesState, START, END
# from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import BaseMessage
from typing import List, TypedDict, Literal

#-----------------------------------------------------------------
# from langgraph.checkpoint.memory import MemorySaver
class MessagesState(TypedDict):
    messages: List[BaseMessage]

# We need a custom state

class SimpleAgentState(MessagesState):
    """Extends the default MessagesState to add a current answer type 
    to allow us to properly route the messages"""
    response_type: str

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolExecutor, ToolInvocation
import json

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
from workflowLlamaIndex.CRAGLlamaIndexWorkflow import run_workflow
from workflowLlamaIndex.CRAGLlamaIndexWorkflow import CorrectiveRAGLlamIndexWorkflow
# from qdrant_client import QdrantClient
import chainlit as cl
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.config import RunnableConfig
from langchain.schema.runnable import Runnable
from dotenv import load_dotenv
import uuid
from uuid import uuid4

########################################################
# Setup the OpenAI model
llm_chatbot = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
#------------------------------------------------------
llm_retriever=ChatOpenAI(model='gpt-4o', temperature=0, streaming=True)
# Define the tool and wrap it correctly
tool_belt = [Llamarag]
functions = [convert_to_openai_function(t) for t in tool_belt]
tool_executor = ToolExecutor(tool_belt)

#-----------------------------------------
# retriever_tool_node = ToolNode(tool_belt)  # Ensure tools are correctly integrated

# Bind tools to the model and the config
# model = llm_retriever.bind_tools(tool_belt)  # Might need to convert or adapt tools
# model = model.with_config(tags=["final_node"])
#-------------------------------------------------------------------
#################################################################################
# RUNNABLE CHAINS
# Create our runnable chains, depending on the prompt we want to pass forward

main_prompt = """
You are a helpful agent designed to assist the user with information from the uploaded PDF document.

You collaborate with another agent that retrieves information from the uploaded document for you.

When a user asks a question, you will forward the query to the retriever agent. 

You will receive information from the retriever agent with the results of their research
into the user's query. Your task is to repeat it word for word to the user. 

Do not summarize the other agent's answer. 

If the other agent retrieved 'no information found!', politely inform user and that the answer is not available in the provided document.

"""

# Create a chain with the main prompt
main_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", main_prompt
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chat_runnable = main_prompt_template | llm_chatbot


#################################################
retriever_prompt = f"""
You are a helpful and extremely thorough agent designed to retrieve information ONLY from the uploaded PDF document. 
Your task is to provide accurate and relevant information based solely on the contents of this document.

ONLY consult with the below tool(s) to retrieve information from the uploaded document to construct your final answer. 

You have access to the following tool(s) ONLY to consult with:

============================
Llamarag
============================

YOUR TASK:

When a user asks a question, use the Llamarag tool to search for relevant information in the uploaded document.

If the generated response by Llamarag is helpful for the user's question, use that information to return back to the chatbot agent.

You MUST ALWAYS cite your sources within the uploaded document. 

When the user asks for information that is not found in the uploaded document, you MUST answer with: ' no information found!'

Do not provide any information beyond what is contained in the uploaded PDF.

Your sole purpose is to extract and provide information ONLY from the uploaded PDF document.
Never make up or infer information. If the exact information is not in the PDF, say so.
Always cite the specific part of the PDF where you found the information.
If you cannot find relevant information, always respond with 'No relevant information found in the uploaded PDF.'
"""
retriever_prompt_template = ChatPromptTemplate.from_messages(
    [
        "system", retriever_prompt,
        MessagesPlaceholder(variable_name="messages")

    ]

)

retrievable_runnable=retriever_prompt_template | llm_retriever.bind_tools(tool_belt).with_config(tags=["final_node"])

#------------------------------------



#-------------------------------------

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
import asyncio
from uuid import uuid4
from typing import List, Dict, Callable, Any
from langchain.schema import HumanMessage, AIMessage
from realtime.tools import rag_query_handler
import os
import json

client = AsyncOpenAI()

async def setup_openai_realtime():
    """Instantiate and configure the OpenAI Realtime Client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    openai_realtime = RealtimeClient(api_key=os.getenv("OPENAI_API_KEY"))

    conversation_id = str(uuid4())
    
    config = {
        "configurable": {
            "thread_id": conversation_id,
            "checkpoint_ns": f"audio_session_{conversation_id}",
            "checkpoint_id": str(uuid4())
        }
    }
    
    # Save config to session and temp file
    # Set session variables
    cl.user_session.set("track_id", conversation_id)
    cl.user_session.set("config", config)  # Store config by key
    
    
    # Create config file in the realtime folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    realtime_dir = os.path.join(current_dir, 'realtime')
    os.makedirs(realtime_dir, exist_ok=True)
    config_path = os.path.join(realtime_dir, "rag_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    #------------------------------------------------
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
    

    
    cl.user_session.set("LlamaLang", CRAGAgent)
    cl.user_session.set("index", index)
    

    # Let the user know that the system is ready
    msg.content = f"Processing {file.name} done! You can ask questions now!"
    await msg.update()
    
    #--------------------------------------------------------------------------------------------------------
    async def handle_conversation_updated(event):
        item = event.get("item")
        delta = event.get("delta")
        """Currently used to stream audio back to the client."""
        if delta:
            # Only one of the following will be populated for any given event
            if 'audio' in delta:
                audio = delta['audio']  # Int16Array, audio added
                await cl.context.emitter.send_audio_chunk(cl.OutputAudioChunk(mimeType="pcm16", data=audio, track=cl.user_session.get("track_id")))
            if 'transcript' in delta:
                transcript = delta['transcript']  # string, transcript added
                pass
            if 'arguments' in delta:
                arguments = delta['arguments']  # string, function arguments added
                pass
    async def handle_item_completed(item):
        """Used to populate the chat context with transcription once an item is completed."""
        # print(item) # TODO
        pass
    
    async def handle_conversation_interrupt(event):
        """Used to cancel the client previous audio playback."""
        cl.user_session.set("track_id", str(uuid4()))
        await cl.context.emitter.send_audio_interrupt()
        
    async def handle_error(event):
        logger.error(event)

    # Register event handlers
    openai_realtime.on('conversation.updated', handle_conversation_updated)
    openai_realtime.on('conversation.item.completed', handle_item_completed)
    openai_realtime.on('conversation.interrupted', handle_conversation_interrupt)
    openai_realtime.on('error', handle_error)

    cl.user_session.set("openai_realtime", openai_realtime)
    coros = [openai_realtime.add_tool(tool_def, tool_handler) for tool_def, tool_handler in tools]
    await asyncio.gather(*coros)
    

@cl.on_chat_start
async def start():
    
    await cl.Message(
        content="Welcome to the Chainlit x OpenAI realtime example. Press `P` if to talk!"
    ).send()
    await setup_openai_realtime()

@cl.on_message
async def on_message(message: cl.Message):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():      
        await openai_realtime.send_user_message_content([{ "type": 'input_text', "text": message.content }])
    else:
        await cl.Message(content="Please activate voice mode before sending messages!").send()

@cl.on_audio_start
async def on_audio_start():
    try:
        openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
        await openai_realtime.connect()
        logger.info("Connected to OpenAI realtime")
        # TODO: might want to recreate items to restore context
        # openai_realtime.create_conversation_item(item)
        return True
    except Exception as e:
        await cl.ErrorMessage(content=f"Failed to connect to OpenAI realtime: {e}").send()
        return False

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")            
    if openai_realtime.is_connected():
        await openai_realtime.append_input_audio(chunk.data)
    else:
        logger.info("RealtimeClient is not connected")


@cl.on_audio_end
@cl.on_chat_end
@cl.on_stop
async def on_end():
    """Clean up when the chat or audio ends."""
    openai_realtime = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.disconnect()

