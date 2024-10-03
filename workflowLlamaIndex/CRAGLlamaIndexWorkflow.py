

from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from llama_index.core import (
    VectorStoreIndex,
    Document,
    SummaryIndex,
    PromptTemplate,
)
from llama_index.utils.workflow import draw_all_possible_flows

from llama_index.core.query_pipeline import QueryPipeline
from llama_index.llms.openai import OpenAI
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.core.base.base_retriever import BaseRetriever
#---------------------------
from IPython.display import Markdown, display
from llama_index.core import SimpleDirectoryReader
from pyvis.network import Network
import asyncio
#---------------------------
import os
from dotenv import load_dotenv
# #----------------------------
load_dotenv()

# VERSION = '1.0_rc3'
# os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGCHAIN_PROJECT"] + f" - v. {VERSION}"

openai_api_key = os.getenv('OPENAI_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')
#----------------------------------------------
# Events as subclasses: Pydantic Type variables
#-----------------------------------------------
# The Workflow Events
class PrepEvent(Event):
    """Prep event (prepares for retrieval)."""
    pass

class RetrieveEvent(Event):
    """Retrieve event (gets retrieved nodes)."""

    retrieved_nodes: list[NodeWithScore]

class RelevanceEvalEvent(Event):
    """Relevance evaluation event (gets results of relevance evaluation)."""
    relevant_results: list[str]

class TextExtractEvent(Event):
    """Text extract event. Extracts relevant text and concatenates."""
    relevant_text: str

class QueryEvent(Event):
    """Query event. Queries given relevant text and search text."""
    relevant_text: str
    search_text: str
#---------------------------------
# PROMPT PREPERATION PIPELINE USES
#---------------------------------
DEFAULT_RELEVANCY_PROMPT_TEMPLATE = PromptTemplate(
template="""As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

Retrieved Document:
-------------------
{context_str}

User Question:
--------------
{query_str}

Evaluation Criteria:
- Consider whether the document contains keywords or topics related to the user's question.
- The evaluation should not be overly stringent; the primary objective is to identify and filter out clearly irrelevant retrievals.

Decision:
- Assign a binary score to indicate the document's relevance.
- Use 'yes' if the document is relevant to the question, or 'no' if it is not.

Please provide your binary score ('yes' or 'no') below to indicate the document's relevance to the user question."""
)
DEFAULT_TRANSFORM_QUERY_TEMPLATE = PromptTemplate(
template="""Your task is to refine a query to ensure it is highly effective for retrieving relevant search results. \n
Analyze the given input to grasp the core semantic intent or meaning. \n
Original Query:
\n ------- \n
{query_str}
\n ------- \n
Your goal is to rephrase or enhance this query to improve its search performance. Ensure the revised query is concise and directly aligned with the intended search objective. \n
Respond with the optimized query only:"""
)

#

class CorrectiveRAGLlamIndexWorkflow(Workflow):
#-------------------------
# STEPS
#--------------------------
#1- Ingest and Process Data
    @step
    async def ingest(self, ctx: Context, ev: StartEvent) ->  StopEvent | None:
        """Ingest step (for ingesting docs and initializing index)."""
        documents: list[Document] | None = ev.get("documents")

        if documents is None:
            return None

        index = VectorStoreIndex.from_documents(documents)

        return StopEvent(result=index)
    #---------------------------------------------------
    #2- Prepare the Pipeline for Use
    #---------------------------------------------------
    # This setp preapares two pipelines for the downstream applications:
    #2-1: relavancy pipeline
    #2-2: transform_query pipeline

    # ------------------------------------------------------------------------------------
    @step
    async def prepare_for_retrieval(
        self, ctx: Context, ev: StartEvent
    ) ->  PrepEvent | None:
        """Prepare for retrieval."""

        query_str: str | None = ev.get("query_str")
        retriever_kwargs: dict | None = ev.get("retriever_kwargs", {})

        if query_str is None:
            return None

        tavily_ai_apikey: str | None = ev.get("tavily_ai_apikey")
        index = ev.get("index")

        llm = OpenAI(model="gpt-4o-mini")
        await ctx.set("relevancy_pipeline", QueryPipeline(
            chain=[DEFAULT_RELEVANCY_PROMPT_TEMPLATE, llm]
        ))
        await ctx.set("transform_query_pipeline", QueryPipeline(
            chain=[DEFAULT_TRANSFORM_QUERY_TEMPLATE, llm]
        ))

        await ctx.set("llm", llm)
        await ctx.set("index", index)
        await ctx.set("tavily_tool", TavilyToolSpec(api_key=tavily_ai_apikey))

        await ctx.set("query_str", query_str)
        await ctx.set("retriever_kwargs", retriever_kwargs)

        return PrepEvent()

    #-------------------------------------------------
    #3- Retrieving Context
    #-------------------------------------------------
    @step
    async def retrieve(
        self, ctx: Context, ev: PrepEvent
    ) ->  RetrieveEvent | None:
        """Retrieve the relevant nodes for the query."""
        query_str = await ctx.get("query_str")
        retriever_kwargs = await ctx.get("retriever_kwargs")

        if query_str is None:
            return None

        index = await ctx.get("index", default=None)
        tavily_tool = await ctx.get("tavily_tool", default=None)

        if not (index or tavily_tool):
            raise ValueError(
                "Index and tavily tool must be constructed. Run with 'documents' and 'tavily_ai_apikey' params first."
            )

        retriever: BaseRetriever = index.as_retriever(
            **retriever_kwargs
        )
        result = retriever.retrieve(query_str)
        await ctx.set("retrieved_nodes", result)
        await ctx.set("query_str", query_str)
        return RetrieveEvent(retrieved_nodes=result)
    #-----------------------------------
    #4- Evaluate Context
    #-----------------------------------
    @step
    async def eval_relevance(
        self, ctx: Context, ev: RetrieveEvent
    ) -> RelevanceEvalEvent:
        """Evaluate relevancy of retrieved documents with the query."""
        retrieved_nodes = ev.retrieved_nodes
        query_str = await ctx.get("query_str")
        relevancy_pipe= await ctx.get("relevancy_pipeline")

        relevancy_results = []
        for node in retrieved_nodes:
            relevancy = relevancy_pipe.run(
                context_str=node.text, query_str=query_str
            )
            relevancy_results.append(relevancy.message.content.lower().strip())
            # Then, the relavancy pipeline runs withthe prompt.
        await ctx.set("relevancy_results", relevancy_results)
        return RelevanceEvalEvent(relevant_results=relevancy_results)
    # List of str:yes/no
    #---------------------------------
    #5- Exrtact Relavat Text
    #---------------------------------
    @step
    async def extract_relevant_texts(
        self, ctx: Context, ev: RelevanceEvalEvent) -> TextExtractEvent:
        """Extract relevant texts from retrieved documents."""

        retrieved_nodes=await ctx.get("retrieved_nodes")
        relevancy_results= ev.relevant_results

        relevant_texts=[
            retrieved_nodes[i].text
            for i, result in enumerate(relevancy_results)
            if result=='yes'
        ]
        result='\n'.join(relevant_texts)


        return TextExtractEvent(relevant_text=result)
    #--------------------------------------
    #6- Transform Query
    #-------------------------------------
    @step
    async def transform_query_pipeline(
        self, ctx: Context, ev: TextExtractEvent) -> QueryEvent:
        """Search the transformed query with Tavily API."""

        relevant_text= ev.relevant_text
        relevancy_results=await ctx.get("relevancy_results")
        query_str= await ctx.get("query_str")

        #if any non-relevant contexts, transform our user's query and query our external search tool.
        if "no" in relevancy_results:
            qp= await ctx.get("transform_query_pipeline")
            transformed_query_str=qp.run(query_str=query_str).message.content
            # Conduct a search with the transformed query string and collect the results.
            search_tool=await ctx.get("tavily_tool")
            search_results= search_tool.search(transformed_query_str, max_results=5)

            search_text="\n".join([result.text for result in search_results])
        else:
            search_text=""

        return QueryEvent(relevant_text=relevant_text, search_text=search_text)
    #------------------------------------
    #7 fire off the final query + context package to our LLM for a response!
    #-----------------------------------
    @step
    async def query_result(
        self, ctx: Context, ev: QueryEvent
    ) -> StopEvent:
        """Get results from relevant texts."""
        relevant_text=ev.relevant_text
        search_text= ev.search_text
        query_str=await ctx.get("query_str")

        documents=[Document(text=relevant_text +"\n"+search_text)]
        index=SummaryIndex.from_documents(documents=documents)
        query_engine=index.as_query_engine()
        result=query_engine.query(query_str)
        return StopEvent(result=result)

    
async def run_workflow(query_str: str):
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))

# Combine it with the relative path to the data directory
    data_path = os.path.join(dir_path, "data")
    documents = SimpleDirectoryReader(data_path).load_data()
    CRAGAgent = CorrectiveRAGLlamIndexWorkflow()
    print("\nWorkflow initialized.\n")
    try:
        index = await CRAGAgent.run(documents=documents)
        print("Workflow run completed.\n")

        # draw_all_possible_flows(
        # CRAGAgent, filename="crag_workflow.html")

        response = await CRAGAgent.run(
            query_str=query_str,
            index=index,
            tavily_ai_apikey=tavily_api_key,
        )
        print()
        return str(response)
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        raise

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(run_workflow("where the hallucination is more harmful ?"))

