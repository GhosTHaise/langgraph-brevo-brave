"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
import os
from typing import TypedDict

from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Annotated, Sequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage ,SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from agent.tools.brevo import send_email,email_instructions

class AgentState(TypedDict):
   messages : Annotated[Sequence[BaseMessage] , add_messages]

generative_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
#generative_model = ChatGroq(model="llama-3.1-8b-instant")
@tool
def generate_email_body(subject: str = "", prompt: str = "") -> dict:
    """
    Generate a beautiful HTML email body based on the given context or description.
    Returns a dict so other tools can reuse the result easily.
    """
    full_context = prompt or subject
    if not full_context.strip():
        raise ValueError("generate_email_body requires a non-empty subject or prompt.")

    system_prompt = SystemMessage(
        content=(
            "You are a professional HTML email writer. "
            "Create a well-formatted, attractive, and mobile-friendly HTML email body "
            "based on the given context. "
            "Do NOT include any subject or recipient info. Output only valid HTML content."
        )
    )
    human_message = HumanMessage(content=full_context)

    response = generative_model.invoke([system_prompt, human_message])

    print("=>>>>",response.content)
    # ✅ Return only the content (the actual HTML), not the whole message
    return {"body": response.content}



    
tools = [send_email, generate_email_body]
#model = ChatGroq(model="llama-3.1-8b-instant").bind_tools(tools)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

system_prompt = SystemMessage(content="""
            You are **AgentIA**, an intelligent assistant designed to interact with and utilize an LLM model to process user requests.

            Your purpose is to:
            - Understand the user’s intent.
            - Call the LLM model when reasoning, drafting, or generating new content is required.
            - Clearly display or return the model’s responses in a well-structured way.

            Guidelines:
            - When the user requests a generation, summarization, or completion, call the LLM model and provide the output.
            - When the user asks to do an action, call the relevant tool (e.g., 'update') with the full modified result.
            - Always show the current state or output after any modification or LLM call.
            
            Tools: 
            - generate_email_body: creates a formatted HTML email body.
            - send_email: Sends an email using Brevo .
            
            {email_instructions}
    """)

def call_model(state: State) -> AgentState:
    """Process input and returns output."""
    
        
    all_messages = [system_prompt] + list(state["messages"])
    response = model.invoke(all_messages)
    
    return {"messages" : list(state["messages"]) + [response]}
    
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

def should_continue(state: AgentState) -> bool:
    """Determine if the agent should continue and last message contains tools calls"""
    
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

# Define the graph
graph = (
    StateGraph(AgentState)
    .add_node("llm",call_model)
    .add_node("tools",ToolNode(tools))
    .add_edge("__start__", "llm")
    .add_conditional_edges(
        "llm",
        should_continue,
        {True: "tools" , False: "__end__"}
    )
    .add_edge("tools", "llm")
    .compile(name="LLM -> Brevo Graph")
)
