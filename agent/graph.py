"""LangGraph agent that proposes GEMM configurations, benchmarks them, and iterates.

Loop:  propose -> evaluate -> analyze -> (propose again | stop)
The proposer uses an LLM when OPENAI_API_KEY is set, and a simple search otherwise,
so the pipeline can always run end to end.
"""
import itertools
import json
import os
import uuid
from typing import TypedDict

from langgraph.graph import END, StateGraph

from bench.run_benchmark import run, validate
from agent import store

SEARCH_SPACE = [{"kernel": 0}] + \
    [{"kernel": 1, "tile": t} for t in (8, 16, 32)] + \
    [{"kernel": 2, "tile": t, "coarsen": c} for t, c in itertools.product((16, 32, 64), (2, 4, 8, 16))]


class State(TypedDict):
    run_id: str
    iteration: int
    max_iters: int
    candidate: dict
    reasoning: str
    history: list[dict]
    best: dict | None


def _key(cfg: dict) -> tuple:
    return (cfg["kernel"], cfg.get("tile", 32), cfg.get("coarsen", 1))


def _llm_propose(history: list[dict]) -> tuple[dict, str] | None:
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from openai import OpenAI
    summary = [{k: h.get(k) for k in ("kernel", "tile", "coarsen", "status", "gflops",
                                       "pct_of_cublas", "error")} for h in history]
    prompt = (
        "You are optimizing a CUDA SGEMM kernel on an NVIDIA GPU. Kernels: 0 = naive, "
        "1 = shared-memory tiling (param tile in 8,16,32,64), 2 = tiling + thread coarsening "
        "(params tile and coarsen; tile % coarsen == 0; tile*tile/coarsen <= 1024 threads; "
        "2*tile*tile*4 bytes of shared memory <= 48KB). "
        f"Experiments so far: {json.dumps(summary)}. "
        "Propose ONE untried configuration most likely to improve GFLOPS. Reply as JSON: "
        '{"kernel": int, "tile": int, "coarsen": int, "reasoning": "one sentence"}')
    reply = OpenAI().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"})
    data = json.loads(reply.choices[0].message.content)
    cfg = {"kernel": int(data["kernel"]), "tile": int(data.get("tile", 32)),
           "coarsen": int(data.get("coarsen", 1))}
    return cfg, data.get("reasoning", "")


def propose(state: State) -> dict:
    tried = {_key(h) for h in state["history"]}
    suggestion = None
    try:
        suggestion = _llm_propose(state["history"])
    except Exception as exc:  # fall back to search if the LLM call fails
        print(f"LLM proposal failed ({exc}); using search")
    if suggestion and _key(suggestion[0]) not in tried and validate(suggestion[0]) is None:
        cfg, reason = suggestion
    else:
        untried = [c for c in SEARCH_SPACE if _key(c) not in tried and validate(c) is None]
        cfg = untried[0] if untried else SEARCH_SPACE[0]
        reason = "search: next untried valid configuration"
    return {"candidate": cfg, "reasoning": reason}


def evaluate(state: State) -> dict:
    result = run(state["candidate"])
    store.log(state["run_id"], state["iteration"], result, state["reasoning"])
    print(f"[{state['iteration']}] {state['candidate']} -> {result.get('status')} "
          f"{result.get('gflops', '-')} GFLOPS ({result.get('pct_of_cublas', '-')}% of cuBLAS)")
    return {"history": state["history"] + [result], "iteration": state["iteration"] + 1}


def analyze(state: State) -> dict:
    ok = [h for h in state["history"] if h.get("status") == "ok"]
    best = max(ok, key=lambda h: h["gflops"]) if ok else None
    return {"best": best}


def should_continue(state: State) -> str:
    return "propose" if state["iteration"] < state["max_iters"] else END


def build_graph():
    g = StateGraph(State)
    g.add_node("propose", propose)
    g.add_node("evaluate", evaluate)
    g.add_node("analyze", analyze)
    g.set_entry_point("propose")
    g.add_edge("propose", "evaluate")
    g.add_edge("evaluate", "analyze")
    g.add_conditional_edges("analyze", should_continue)
    return g.compile()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=12)
    args = parser.parse_args()

    final = build_graph().invoke({
        "run_id": uuid.uuid4().hex[:8], "iteration": 0, "max_iters": args.iters,
        "candidate": {}, "reasoning": "", "history": [], "best": None,
    }, {"recursion_limit": args.iters * 4 + 10})
    print("\nBest configuration:", json.dumps(final["best"], indent=2))
