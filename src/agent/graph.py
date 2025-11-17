"""graph.py"""
import os
import getpass
from typing import Literal, Annotated, Sequence
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage

# Import our tools
from agent.tools.tools import send_email, generate_email_body

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# --- Configuration ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define the tools list
tools = [generate_email_body, send_email]

# Initialize Model
# We bind tools here. critically, we do NOT set tool_choice="any".
# This allows the model to stop calling tools when it's done.
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# --- Nodes ---

def call_model(state: AgentState):
    messages = state["messages"]
    
    # System prompt to enforce behavior
    system_prompt = SystemMessage(content=(
        "You are an email assistant. "
        "Process: \n"
        "1. If the user asks to send an email, ALWAYS generate the body first using `generate_email_body`.\n"
        "2. Take the HTML output from step 1 and pass it into `send_email`.\n"
        "3. Once `send_email` returns 'SUCCESS', output a final text confirmation to the user and STOP."
    ))
    
    # Prepend system prompt if not present (or just let the model handle context)
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}

def call_tool(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_results = []
    
    # Iterate over all tool calls (models might output multiple parallel calls)
    for tool_call in last_message.tool_calls:
        action = tool_call
        tool_name = action["name"]
        tool_args = action["args"]
        tool_call_id = action["id"] # Critical for syncing with LLM
        
        print(f"⚙️ Executing Tool: {tool_name}")
        
        # Router for tools
        if tool_name == "generate_email_body":
            result = generate_email_body.invoke(tool_args)
        elif tool_name == "send_email":
            result = send_email.invoke(tool_args)
        else:
            result = "Error: Tool not found."
            
        # Create the proper ToolMessage
        tool_results.append(ToolMessage(
            tool_call_id=tool_call_id,
            content=str(result),
            name=tool_name
        ))
        
    return {"messages": tool_results}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM made tool calls, go to the tool node
    if last_message.tool_calls and last_message.tool_calls.length > 0:
        return "tools"
    # Otherwise, stop
    return "__end__"

# --- Graph Definition ---
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)
workflow.add_edge("tools", "agent") # Loop back to agent to process tool result

graph = workflow.compile()

# --- Test Execution ---
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    print("Starting Graph...")
    user_input = "Send an email to ghostrex2@gmail.com about our meeting on Friday. Say we need to discuss the roadmap."
    
    # Stream the output to see steps
    for event in graph.stream({"messages": [HumanMessage(content=user_input)]}):
        for key, value in event.items():
            print(f"\n--- Node: {key} ---")
            # Uncomment to see full state details
            # print(value)