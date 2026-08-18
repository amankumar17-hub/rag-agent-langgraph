from dotenv import load_dotenv
load_dotenv()

from kensho_agent import llm, tools, AgentState, llm_node, should_continue, run_agent
from kensho_rag import search_sec_filing
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# add rag tool to existing tools
all_tools = tools + [search_sec_filing]

# rebuild graph with all 4 tools
tool_node = ToolNode(all_tools)
llm_with_all_tools = llm.bind_tools(all_tools)

# redefine llm_node to use updated llm
def llm_node_full(state: AgentState):
    response = llm_with_all_tools.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("llm", llm_node_full)
graph.add_node("tools", tool_node)
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")
agent = graph.compile(checkpointer = None)

# updated system message
system = SystemMessage(content=(
    "You are a financial analysis assistant for Apple Inc. "
    "You have access to four tools:\n"
    "- get_stock_price: for current price, volume, market cap\n"
    "- get_company_info: for sector and industry\n"
    "- get_financial_metrics: for P/E, EPS, 52-week range\n"
    "- search_sec_filing: for qualitative info from Apple's 10-K\n"
    "Rules:\n"
    "1. Call each tool at most once per question\n"
    "2. Do not repeat a search_sec_filing call with a similar query\n"
    "3. Once you have retrieved any information, use it to answer immediately\n"
    "4. Do not call tools more than 5 times total per question\n"
    "5. If any tool is not giving the required information then just dont call it again and try to call another tool to fetch relevant or remaining info\n"
    "Provide your final answer as soon as you have relevant information."
))

def run_agent_full(question: str):
    print(f"\nQuestion: {question}")
    print("-" * 50)
    result = agent.invoke({
        "messages": [system, HumanMessage(content=question)]},
        config={"recursion_limit": 10})
    
    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"[{msg_type}] → calling tools: {[t['name'] for t in msg.tool_calls]}")
        elif hasattr(msg, "name") and msg.name:
            print(f"[ToolMessage:{msg.name}] → {msg.content[:100]}...")
        else:
            print(f"[{msg_type}] → {msg.content}")
    print("-" * 50)

# test
# run_agent_full("What are Apple's main revenue sources according to their 10-K, and how does that reflect in their current stock price?")
run_agent_full("What risk factors does Apple mention in their SEC filing, and how does their beta reflect that risk?")