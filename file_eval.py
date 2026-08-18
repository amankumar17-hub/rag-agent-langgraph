# 05_eval.py
from dotenv import load_dotenv
load_dotenv()

from kensho_agent import run_agent, agent, SYSTEM_MESSAGE
from langchain_core.messages import HumanMessage
import json

# ── Step 1: Gold dataset ───────────────────────────────────────────────────────
# Each entry has:
# - question: what we ask the agent
# - expected_keywords: words that MUST appear in a correct answer
# - expected_tools: tools the agent should call to answer this
EVAL_DATASET = [
    {
        "id": "q1",
        "question": "What is Apple's current stock price?",
        "expected_keywords": ["AAPL", "$", "market cap", "volume"],
        "expected_tools": ["get_stock_price"],
        "must_not_contain": ["I don't know", "unable to", "error"],
    },
    {
        "id": "q2",
        "question": "What sector is Apple in?",
        "expected_keywords": ["Technology", "sector", "industry"],
        "expected_tools": ["get_company_info"],
        "must_not_contain": ["I don't know", "unable to"],
    },
    {
        "id": "q3",
        "question": "What is Apple's P/E ratio and what does it mean?",
        "expected_keywords": ["P/E", "earnings", "price"],
        "expected_tools": ["get_financial_metrics"],
        "must_not_contain": ["I don't know", "unable to"],
    },
    {
        "id": "q4",
        "question": "Is Apple trading closer to its 52-week high or low right now?",
        "expected_keywords": ["52-week", "high", "low", "current"],
        "expected_tools": ["get_financial_metrics", "get_stock_price"],
        "must_not_contain": ["I don't know", "unable to"],
    },
    {
        "id": "q5",
        "question": "Should I buy Apple stock right now?",
        "expected_keywords": ["P/E", "price", "risk"],
        "expected_tools": ["get_stock_price", "get_financial_metrics"],
        "must_not_contain": ["I don't know"],
    },
    {
        "id": "q6",
        "question": "What is the weather like today?",          # irrelevant question
        "expected_keywords": [],
        "expected_tools": [],                                   # should call NO tools
        "must_not_contain": ["$", "AAPL", "stock"],
    },
    {
        "id": "q7",
        "question": "What is Apple's beta and what does it tell us about risk?",
        "expected_keywords": ["beta", "risk", "market", "volatility"],
        "expected_tools": ["get_financial_metrics"],
        "must_not_contain": ["I don't know"],
    },
]

# ── Step 2: Scoring functions ──────────────────────────────────────────────────
def score_keywords(response: str, expected_keywords: list, must_not_contain: list = []) -> dict:
    response_lower = response.lower()
    
    # keywords that must appear
    keyword_results = {}
    for kw in expected_keywords:
        keyword_results[kw] = kw.lower() in response_lower

    # phrases that must NOT appear
    negative_results = {}
    for phrase in must_not_contain:
        negative_results[phrase] = phrase.lower() not in response_lower  # True = good

    all_checks = list(keyword_results.values()) + list(negative_results.values())
    score = sum(all_checks) / len(all_checks) if all_checks else 1.0

    return {
        "score": score,
        "keyword_hits": keyword_results,
        "negative_hits": negative_results
    }

def score_tool_calls(messages: list, expected_tools: list) -> dict:
    called_tools = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for t in msg.tool_calls:
                called_tools.append(t["name"])

    # if no tools expected, score is 1.0 if no tools called, 0.0 if any called
    if not expected_tools:
        score = 1.0 if len(called_tools) == 0 else 0.0
        return {
            "score": score,
            "detail": {},
            "redundant_calls": {},
            "total_tool_calls": len(called_tools)
        }

    from collections import Counter
    results = {tool: tool in called_tools for tool in expected_tools}
    redundant = {k: v for k, v in Counter(called_tools).items() if v > 1}
    score = sum(results.values()) / len(results)
    return {
        "score": score,
        "detail": results,
        "redundant_calls": redundant,
        "total_tool_calls": len(called_tools)
    }


def score_latency(elapsed: float) -> dict:
    """Score based on response time. Under 5s = good, under 10s = ok, over 10s = bad."""
    if elapsed < 5:
        score = 1.0
        label = "good"
    elif elapsed < 10:
        score = 0.5
        label = "acceptable"
    else:
        score = 0.0
        label = "slow"
    return {"score": score, "elapsed": round(elapsed, 2), "label": label}


# ── Step 3: Run eval ───────────────────────────────────────────────────────────
def run_eval(dataset: list) -> list:
    results = []

    for item in dataset:
        print(f"\nEvaluating {item['id']}: {item['question']}")
        print("-" * 50)

        import time
        start = time.time()

        try:
            result = agent.invoke(
                {"messages": [SYSTEM_MESSAGE, HumanMessage(content=item["question"])]},
                config={"recursion_limit": 10}
            )
            elapsed = time.time() - start

            # extract final response
            final_response = ""
            for msg in result["messages"]:
                if not hasattr(msg, "tool_calls") and not hasattr(msg, "name"):
                    final_response = msg.content
            
            # if last message is the final answer
            final_response = result["messages"][-1].content

            # score
            keyword_score = score_keywords(
                                            final_response,
                                            item["expected_keywords"],
                                            item.get("must_not_contain", [])
                                        )
            tool_score     = score_tool_calls(result["messages"], item["expected_tools"])
            latency_score  = score_latency(elapsed)

            overall = (keyword_score["score"] + tool_score["score"] + latency_score["score"]) / 3

            eval_result = {
                "id":             item["id"],
                "question":       item["question"],
                "response":       final_response[:200],
                "keyword_score":  keyword_score,
                "tool_score":     tool_score,
                "latency_score":  latency_score,
                "overall_score":  round(overall, 2),
            }

            results.append(eval_result)
            print(f"Overall score: {overall:.2f}")
            print(f"Keywords:      {keyword_score['score']:.2f} {keyword_score['keyword_hits']}")
            print(f"Tools:         {tool_score['score']:.2f} {tool_score['detail']}")
            print(f"Latency:       {latency_score['elapsed']}s ({latency_score['label']})")

        except Exception as e:
            print(f"Failed: {str(e)}")
            results.append({"id": item["id"], "error": str(e), "overall_score": 0})

    return results


# ── Step 4: Report ─────────────────────────────────────────────────────────────
def print_report(results: list):
    print("\n" + "=" * 60)
    print("EVAL REPORT")
    print("=" * 60)
    

    scores = [r["overall_score"] for r in results if "error" not in r]
    avg = sum(scores) / len(scores)

    print(f"\n{'ID':<6} {'Overall':>8} {'Keywords':>10} {'Tools':>8} {'Latency':>10}")
    print("-" * 50)

    for r in results:
        if "error" in r:
            print(f"{r['id']:<6} ERROR: {r['error']}")
            continue
        print(
            f"{r['id']:<6} "
            f"{r['overall_score']:>8.2f} "
            f"{r['keyword_score']['score']:>10.2f} "
            f"{r['tool_score']['score']:>8.2f} "
            f"{r['latency_score']['elapsed']:>9.2f}s"
        )
        # print keyword detail
        print(f"       keyword hits:  {r['keyword_score']['keyword_hits']}")
        print(f"       negative hits: {r['keyword_score']['negative_hits']}")
        print(f"       tools called:  {r['tool_score']['detail']}")
        if r['tool_score']['redundant_calls']:
            print(f"       redundant:     {r['tool_score']['redundant_calls']}")
        print(f"       total tools called: {r['tool_score']['total_tool_calls']}")


    print("-" * 50)
    print(f"{'AVERAGE':<6} {avg:>8.2f}")
    print(f"\nTotal questions: {len(results)}")
    print(f"Passed (>0.7):   {sum(1 for s in scores if s > 0.7)}")
    print(f"Failed (<0.5):   {sum(1 for s in scores if s < 0.5)}")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to eval_results.json")
if __name__ == "__main__":
    results = run_eval(EVAL_DATASET)
    print_report(results)