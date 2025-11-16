"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
import os
from typing import Literal, Optional

from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Annotated, Sequence
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage ,SystemMessage,HumanMessage, AIMessage
from langgraph.graph.message import add_messages
#from langgraph.prebuilt import ToolNode
from agent.tools.brevo import send_email,generate_email_body
from agent.configuration import Configuration
from langchain_core.runnables import RunnableConfig
from agent.utils import init_model

class AgentState(TypedDict):
   messages : Annotated[Sequence[BaseMessage] , add_messages]
   context: dict
   last_tool_output: dict | None
   
tools = [send_email, generate_email_body]
#model = ChatGroq(model="openai/gpt-oss-20b").bind_tools(tools)
#model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

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

def call_model(state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
    context = state.get("context", {})
    context_message = (
        f"Here is the current context:\n{context}\n"
        "Use it to make informed decisions or chain tool results."
    )

    raw_model = init_model(config)
    model = raw_model.bind_tools(tools, tool_choice="any")
    
    all_messages = [system_prompt, HumanMessage(content=context_message)] + list(state["messages"])
    response = model.invoke(all_messages)

    return {
        "messages": list(state["messages"]) + [response],
        "context": context,
        "last_tool_output": state.get("last_tool_output"),
    }

    
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

def call_tool(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]

    # No tool call? stop here.
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return state

    tool_call = last_msg.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # Find the tool function
    tool_fn = next(t for t in tools if t.name == tool_name)
    result = tool_fn.invoke(tool_args)
    print(f"Tool `{tool_name}` executed successfully.\nResult: {result}")
    # Always append an AIMessage with NO tool calls inside
    ai_msg = AIMessage(
        content=f"Tool `{tool_name}` executed successfully.\nResult: {result.message}",
        tool_calls=[]        # <==== VERY IMPORTANT
    )

    return {
        "messages": list(state["messages"]) + [ai_msg],
        "context": {**state.get("context", {}), tool_name: result},
        "last_tool_output": result if isinstance(result, dict) else {"output": str(result)},
    }


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determine if the agent should continue and last message contains tools calls"""
    
    result = state["messages"][-1]
    shouldContinue = hasattr(result, "tool_calls") and len(result.tool_calls) > 0
    return "tools" if shouldContinue else "__end__"

# Define the graph
graph = (
    StateGraph(AgentState, context_schema=Configuration)
    .add_node("llm",call_model)
    .add_node("tools", call_tool)
    .add_edge("__start__", "llm")
    .add_conditional_edges(
        "llm",
        should_continue
    )
    .add_edge("tools", "llm")
    .compile(name="LLM -> Brevo Graph")
)
