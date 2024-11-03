import yfinance as yf
import chainlit as cl
import plotly
from langchain.schema import HumanMessage, AIMessage

from langchain_core.messages import AIMessageChunk, HumanMessage
from typing import List, Tuple, Callable, Dict, Any
from chainlit.logger import logger

query_stock_price_def = {
    "name": "query_stock_price",
    "description": "Queries the latest stock price information for a given stock symbol.",
    "parameters": {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "The stock symbol to query (e.g., 'AAPL' for Apple Inc.)"
        },
        "period": {
          "type": "string",
          "description": "The time period for which to retrieve stock data (e.g., '1d' for one day, '1mo' for one month)"
        }
      },
      "required": ["symbol", "period"]
    }
}

async def query_stock_price_handler(symbol, period):
    """
    Queries the latest stock price information for a given stock symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            return {"error": "No data found for the given symbol."}
        return hist.to_json()
 
    except Exception as e:
        return {"error": str(e)}

query_stock_price = (query_stock_price_def, query_stock_price_handler)

draw_plotly_chart_def = {
    "name": "draw_plotly_chart",
    "description": "Draws a Plotly chart based on the provided JSON figure and displays it with an accompanying message.",
    "parameters": {
      "type": "object",
      "properties": {
        "message": {
          "type": "string",
          "description": "The message to display alongside the chart"
        },
        "plotly_json_fig": {
          "type": "string",
          "description": "A JSON string representing the Plotly figure to be drawn"
        }
      },
      "required": ["message", "plotly_json_fig"]
    }
}

async def draw_plotly_chart_handler(message: str, plotly_json_fig):
    fig = plotly.io.from_json(plotly_json_fig)
    elements = [cl.Plotly(name="chart", figure=fig, display="inline")]

    await cl.Message(content=message, elements=elements).send()
    
draw_plotly_chart = (draw_plotly_chart_def, draw_plotly_chart_handler)


#tools = [query_stock_price, draw_plotly_chart]

# New RAG tool definition
rag_query_def = {
    "name": "rag_query",
    "description": "Queries the RAG system with user questions about uploaded documents",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's question about the uploaded document"
            }
        },
        "required": ["query"]
    }
}

 
#-------------------------------------------------

async def rag_query_handler(query: str) -> str:
    """Handles RAG queries using the configured CRAG system."""
    import json
    import os
    from langchain_core.runnables.config import RunnableConfig

    # Load config and initialize CRAG system
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "rag_config.json")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            cl.user_session.set("config", config)
            AgentCRAG = cl.user_session.get("LlamaLang")
            index = cl.user_session.get("index")
    except FileNotFoundError:
        logger.error("Config file not found")
        return "Configuration not found. Please initialize the system first."

    if not AgentCRAG or not index:
        logger.warning("CRAG system or index not initialized. Please upload a document first.")
        return "CRAG system or index not initialized. Please upload a document first."

    logger.info(f"Processing CRAG query: {query}")

    try:
        response = await AgentCRAG.run(
            query_str=query,
            index=index,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        )
        
        # if response is None or response == "":
        #     logger.warning("Empty response generated from CRAG system.")
        #     logger.info("Checking index and query for potential issues...")
            
        #     # Log index information
        #     logger.info(f"Index type: {type(index)}")
        #     logger.info(f"Index size: {len(index.index_struct.index.docstore._dict)}")
            
        #     # Log query information
        #     logger.info(f"Query: {query}")
            
        #     # Check if the query is in the index
        #     query_vector = index.index_struct.index.embed_model.get_agg_embedding_from_queries([query])
        #     query_results = index.index_struct.query(query_vector)
        #     logger.info(f"Query results: {query_results}")
            
        #     # return "No information found in the provided document."
        # else:
        await cl.Message(content=str(response)).send()
        return str(response)
        

    except Exception as e:
        logger.exception(f"Error processing CRAG query: {str(e)}")
        return f"Error processing query: {str(e)}"
        # return str(response)
   


# Tool configurations
# query_stock_price = (query_stock_price_def, query_stock_price_handler)
# draw_plotly_chart = (draw_plotly_chart_def, draw_plotly_chart_handler)
query_rag = (rag_query_def, rag_query_handler)

# Combined tools list
tools: List[Tuple[Dict[str, Any], Callable]] = [
    query_stock_price,
    draw_plotly_chart,
    query_rag
]