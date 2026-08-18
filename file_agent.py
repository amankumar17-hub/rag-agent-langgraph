from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import operator


# ── State ──────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price, volume and market cap for a given ticker symbol."""
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        return (
            f"Ticker: {ticker.upper()}\n"
            f"Current Price: ${info.get('currentPrice', 'N/A')}\n"
            f"Market Cap: ${info.get('marketCap', 'N/A'):,}\n"
            f"Volume: {info.get('volume', 'N/A'):,}\n"
            f"Previous Close: ${info.get('previousClose', 'N/A')}"
        )
    except Exception as e:
        return f"Error fetching price for {ticker}: {str(e)}"


@tool
def get_company_info(ticker: str) -> str:
    """Get company description, sector and industry for a given ticker symbol."""
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        return (
            f"Company: {info.get('longName', 'N/A')}\n"
            f"Sector: {info.get('sector', 'N/A')}\n"
            f"Industry: {info.get('industry', 'N/A')}\n"
            f"Description: {info.get('longBusinessSummary', 'N/A')[:300]}..."
        )
    except Exception as e:
        return f"Error fetching info for {ticker}: {str(e)}"


@tool
def get_financial_metrics(ticker: str) -> str:
    """Get key financial metrics like P/E ratio, EPS, and 52-week range for a given ticker."""
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        return (
            f"Ticker: {ticker.upper()}\n"
            f"P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
            f"EPS: ${info.get('trailingEps', 'N/A')}\n"
            f"52-week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"52-week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
            f"Dividend Yield: {info.get('dividendYield', 'N/A')}\n"
            f"Beta: {info.get('beta', 'N/A')}"
        )
    except Exception as e:
        return f"Error fetching metrics for {ticker}: {str(e)}"


tools = [get_stock_price, get_company_info, get_financial_metrics]


# ── LLM ────────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="openai/gpt-oss-20b")
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")


# ── Nodes ──────────────────────────────────────────────────────────────────────
def llm_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ── Graph ──────────────────────────────────────────────────────────────────────
tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")
agent = graph.compile()


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_MESSAGE = SystemMessage(content=(
    "You are a financial analysis assistant. "
    "You have access to three tools:\n"
    "- get_stock_price: for current price, volume, market cap\n"
    "- get_company_info: for sector and industry\n"
    "- get_financial_metrics: for P/E, EPS, 52-week range\n"
    "Use the right tool for each type of question. "
    "Do not repeat tool calls for data you already have. "
    "Once you have enough information, provide your final answer immediately."
))


# ── Run ────────────────────────────────────────────────────────────────────────
def run_agent(question: str):
    print(f"\nQuestion: {question}")
    print("-" * 50)

    result = agent.invoke({
        "messages": [SYSTEM_MESSAGE, HumanMessage(content=question)]
    })

    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"[{msg_type}] → calling tools: {[t['name'] for t in msg.tool_calls]}")
        elif hasattr(msg, "name") and msg.name:
            print(f"[ToolMessage:{msg.name}] → {msg.content[:100]}...")
        else:
            print(f"[{msg_type}] → {msg.content}")

    print("-" * 50)
    return result


if __name__ == "__main__":
    run_agent("What is the stock price of Apple?")
    run_agent("Compare Google and Microsoft — which has a better P/E ratio?")