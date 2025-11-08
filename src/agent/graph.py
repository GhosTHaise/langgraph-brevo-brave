"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, TypedDict

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict, Annotated, Sequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage , HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
   messages : Annotated[Sequence[BaseMessage] , add_messages]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def call_model(state: State) -> AgentState:
    """Process input and returns output."""
    
    system_prompt = SystemMessage(content="""
            You are **AgentIA**, an intelligent assistant designed to interact with and utilize an LLM model to process user requests.

            Your purpose is to:
            - Understand the user’s intent.
            - Call the LLM model when reasoning, drafting, or generating new content is required.
            - Clearly display or return the model’s responses in a well-structured way.

            Guidelines:
            - When the user requests a generation, summarization, or completion, call the LLM model and provide the output.
            - When the user asks to modify or update data, call the relevant tool (e.g., 'update') with the full modified result.
            - When the user indicates the task is complete, call the 'save' tool.
            - Always show the current state or output after any modification or LLM call.
    """)
        
    all_messages = [system_prompt] + list(state["messages"])
    response = model.invoke(all_messages)
    
    return {"messages" : list(state["messages"]) + [response]}
    
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Define the graph
graph = (
    StateGraph(AgentState)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="New Graph")
)
