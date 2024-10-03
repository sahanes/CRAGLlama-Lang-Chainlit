import getpass
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI


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
llm_chatbot = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#------------------------------------------------------
llm_retriever=ChatOpenAI(model='gpt-4o', temperature=0)
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
# We need a custom state

class SimpleAgentState(MessagesState):
    """Extends the default MessagesState to add a current answer type 
    to allow us to properly route the messages"""
    response_type: str 

#------------------------------------
# Defining our nodes
# ----------------------------------
def chatbot(state: SimpleAgentState):
    last_message = state['messages'][-1]
    
    if isinstance(last_message, HumanMessage):
        response_type = 'user_query'
    else:
        response_type = 'agent_response'

    invoke_input = {'messages': state['messages'], 'response_type': response_type}
    response = chat_runnable.invoke(invoke_input)
    
    print()
    print('response by chat_runnable agent node:\n ', response)
    print()

    return {'messages': state['messages'] + [response], 'response_type': response_type}

#--------------------------------------------
def retriever(state: SimpleAgentState):
    # Run a retrieval query on the conversation state
    
    
    response=retrievable_runnable.invoke(state["messages"])
    
    print("response by retriverable_runneble agent node:\n ", response)
    print()
    # print('response by retreiver:', response)
    # current_response_type=state['response_type']

    output={"messages": state['messages']+[response], "response_type": 'agent_response'}
    # print()
    # print("response by retriverable_runneble agent node: ", output)
    # print()

    return output

async def retriever_tool(state: SimpleAgentState):
    last_message = state["messages"][-1]
    print('Here retriever_tool:', last_message)
    if isinstance(last_message, AIMessage) and last_message.additional_kwargs:
        tool_calls = last_message.additional_kwargs.get('tool_calls', [])
        if tool_calls:
            new_messages=[]
            tool_call = tool_calls[0]
            tool_name = tool_call['function']['name']
            tool_arguments = json.loads(tool_call['function']['arguments'])
            
            action = ToolInvocation(
                tool=tool_name,
                tool_input=tool_arguments,
            )

            response = await tool_executor.ainvoke(action)
            # function_message = FunctionMessage(content=str(response), name=action.tool)
            # return {"messages": state["messages"] + [function_message]}
            # Create a ToolMessage for each tool call
            tool_message = ToolMessage(
                content=str(response),
                tool_call_id=tool_call['id'],
                name=tool_name
            )
            new_messages.append(tool_message)

            return {"messages": state["messages"] + new_messages}
    
    # If no tool call was found or executed, return the original state
    return state
###################################################################
# Define the graph
##################################################################


def rout_query(state: MessagesState) -> Literal["tools", "end"]:
    response_type = state.get('response_type')

    if response_type == 'user_query':
        # Routing to the query retriever
        return 'query'
    else:
        return "end"


def route_tools(state: SimpleAgentState) -> Literal["tools", "success", "no_answer"]:
    if isinstance(state['messages'][-1], AIMessage):
        ai_message = state['messages'][-1]
        if ai_message.additional_kwargs.get('tool_calls'):
            return "tools"
        elif 'no information found' in ai_message.content.lower():
            print('no information found!')
            return "no_answer"
        else:
            return "success"
    return "success"  # Default case if the last message is not an AIMessage

########################################################################

# # Compile the graph with a checkpointer for state persistence
# # checkpointer = MemorySaver()
# graph = workflow.compile()#checkpointer=checkpointer)
workflow = StateGraph(SimpleAgentState)

workflow.add_node("chatbot", chatbot)
workflow.add_node("retriever", retriever)
workflow.add_node("retriever_tools", retriever_tool)

workflow.add_edge(START, "chatbot")

workflow.add_conditional_edges(
    "chatbot",
    rout_query,
    {'query': "retriever", "end": END}
)

workflow.add_conditional_edges(
    "retriever",
    route_tools,
    {"tools": "retriever_tools", "success": "chatbot", "no_answer": END},
)

workflow.add_edge("retriever_tools", "chatbot")

graph = workflow.compile()

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
    checkpointer=MemorySaver()

    ReActgraph = workflow.compile(checkpointer=checkpointer)

    
    cl.user_session.set("LlamaLang", CRAGAgent)
    cl.user_session.set("index", index)
    cl.user_session.set("ReActLangraph", ReActgraph)

    # Let the user know that the system is ready
    msg.content = f"Processing {file.name} done! You can ask questions now!"
    await msg.update()

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
    # AgentCRAG = cl.user_session.get("LlamaLang")
    # index = cl.user_session.get("index")
    reActLang=cl.user_session.get("ReActLangraph")
    config = cl.user_session.get("config")
    msg=cl.Message(content="")



    # response= await AgentCRAG.run(
    
    #     query_str= message.content,
    #     index=index,
    #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # )


    inputs = {"messages": [HumanMessage(content= message.content)]}

    agent_message = cl.Message(content="")
    await agent_message.send()

    async for event in reActLang.astream_events(inputs,
                                            version="v2",
                                            config=config,
                                            ):
        print(event)
        # kind=event['event']
    # async def run_graph():
    # await reActLang.ainvoke(
    #     {"messages": [HumanMessage(content=message.content)]}, 
    # config=config,
    # )
        # return final_state['messages'][-1].content
    #  Get the current event loop; if there isn't one, it will create a new one
    # 
    # await final_state['messages'][-1].content#run_graph()
    
        kind = event["event"]
        # event_name = event.get('name', '')

        if kind == "on_chain_end":
    #         # Extract the final AI message from the event data
    #         messages = event["data"]["output"]
    #         await agent_message.stream_token('\n\n')
    
    # await cl.Message(content=next((msg for msg in reversed(messages) if isinstance(msg, AIMessage) and msg.content), None)).send()
            output = event["data"]["output"]

            # If 'output' is an instance of AIMessage directly
            if isinstance(output, AIMessage):
                final_message = output.content
            # If 'output' is a dictionary and 'messages' is expected to be a list
            await agent_message.stream_token('\n\n')
            # if final_message:
            #     await cl.Message(content=final_message.content).send()
            
    # # Send empty message to stop the little ball from blinking
    await cl.Message(content=str(final_message)).send()#agent_message.send()
    # pip install -U langchain-community

