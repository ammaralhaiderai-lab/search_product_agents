
# ============================================================
# Environment + LLM setup (VS Code / Streamlit Cloud)
# ============================================================
import asyncio
import getpass
import json
import logging
import os
import re
import socket
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, List, Optional, TypedDict
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from tavily import TavilyClient

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

load_dotenv()

# Support both the old Colab key name used in the working notebook
# and the cleaner names used by this Streamlit app.
api_key = (
    os.getenv("OPENROUTER_API_KEY")
    or os.getenv("openrouterkey")
    or ""
)
tavily_api_key = os.getenv("TAVILY_API_KEY") or ""

if not api_key:
    raise ValueError(
        "Missing OPENROUTER_API_KEY. Set it in .env locally or Streamlit Cloud Secrets."
    )

MODEL_NAME = "openai/gpt-5.4-mini"

llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Tavily is preferred. DuckDuckGo remains the fallback implemented by the
# working notebook if Tavily is unavailable.
if tavily_api_key:
    tavily_client = TavilyClient(api_key=tavily_api_key)
    SEARCH_PROVIDER = "tavily"
else:
    tavily_client = None
    SEARCH_PROVIDER = "duckduckgo"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# SOURCE CELL 3
# ============================================================

# Lightweight tracing compatible with the provided notebook
@dataclass
class Span:
    id: str
    name: str
    start_time: float
    level: int
    type: str = "span"
    parent: Optional["Span"] = None
    children: List["Span"] = field(default_factory=list)
    end_time: Optional[float] = None
    input: Any = None
    output: Any = None
    metadata: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    model: Optional[str] = None

_current_span: ContextVar[Optional[Span]] = ContextVar("current_span", default=None)


def print_tree(span: "Span"):
    duration = ((span.end_time or time.time()) - span.start_time) * 1000
    indent = "  " * span.level
    prefix, suffix = ("=== TRACE: ", " ===") if span.level == 0 else ("|-- ", "")
    meta_parts = []
    if span.model:
        meta_parts.append(f"model={span.model}")
    if span.usage.get("total_tokens"):
        meta_parts.append(f"tokens={span.usage['total_tokens']}")
    if "cost_usd" in span.metadata:
        meta_parts.append(f"${span.metadata['cost_usd']:.4f}")
    meta_str = f" [{', '.join(meta_parts)}]" if meta_parts else ""
    type_str = f" [{span.type}]" if span.type != "span" else ""
    print(f"{indent}{prefix}{span.name}{type_str}{suffix} ({duration:.2f}ms){meta_str}")
    for child in span.children:
        print_tree(child)


def _make_span(span_name: str, span_type: str, args, kwargs) -> "Span":
    parent = _current_span.get()
    level = parent.level + 1 if parent else 0
    span = Span(
        id=str(uuid.uuid4())[:8],
        name=span_name,
        type=span_type,
        start_time=time.time(),
        level=level,
        parent=parent,
        input={"args": args, "kwargs": kwargs},
    )
    if parent:
        parent.children.append(span)
    return span


def _finish_span(span: Span):
    span.end_time = time.time()
    if span.level == 0:
        print("\n" + "-" * 64)
        print_tree(span)
        print("-" * 64 + "\n")


def observe(name=None, as_type=None):
    def decorator(func):
        span_name = name if isinstance(name, str) else func.__name__
        span_type = as_type or "span"
        if asyncio.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                span = _make_span(span_name, span_type, args, kwargs)
                token = _current_span.set(span)
                try:
                    result = await func(*args, **kwargs)
                    if span.output is None:
                        span.output = result
                    return result
                except Exception as e:
                    span.output = f"Error: {e}"
                    raise
                finally:
                    _finish_span(span)
                    _current_span.reset(token)
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper
        else:
            def wrapper(*args, **kwargs):
                span = _make_span(span_name, span_type, args, kwargs)
                token = _current_span.set(span)
                try:
                    result = func(*args, **kwargs)
                    if span.output is None:
                        span.output = result
                    return result
                except Exception as e:
                    span.output = f"Error: {e}"
                    raise
                finally:
                    _finish_span(span)
                    _current_span.reset(token)
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper
    if callable(name):
        func, name = name, None
        return decorator(func)
    return decorator


class LangfuseContext:
    def update_current_observation(self, **kwargs):
        span = _current_span.get()
        if not span:
            return
        if isinstance(kwargs.get("usage"), dict):
            span.usage.update(kwargs["usage"])
        if "model" in kwargs:
            span.model = kwargs["model"]
        if isinstance(kwargs.get("metadata"), dict):
            span.metadata.update(kwargs["metadata"])


langfuse_context = LangfuseContext()

# ============================================================
# SOURCE CELL 4
# ============================================================

# LoopDetector: repetition + fuzzy repetition + stagnation
@dataclass
class LoopDetectionResult:
    is_looping: bool
    strategy: str
    message: str
    confidence: float


class LoopDetector:
    def __init__(self, exact_threshold: int = 2, fuzzy_threshold: float = 0.8, stagnation_window: int = 3):
        self.exact_threshold = exact_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.stagnation_window = stagnation_window
        self.tool_history: list[tuple[str, str]] = []
        self.output_history: list[str] = []

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        return len(tokens1 & tokens2) / len(tokens1 | tokens2)

    def check_tool_call(self, tool_name: str, tool_input: str) -> LoopDetectionResult:
        current = (tool_name, tool_input.strip())
        exact_count = sum(
            1 for past_tool, past_input in self.tool_history
            if (past_tool, past_input.strip()) == current
        )
        if exact_count >= self.exact_threshold:
            self.tool_history.append(current)
            return LoopDetectionResult(
                True, "exact",
                f"Exact loop detected: '{tool_name}' repeated with identical arguments.", 1.0,
            )

        recent = self.tool_history[-5:]
        fuzzy_matches = sum(
            1 for past_tool, past_input in recent
            if past_tool == tool_name and self._jaccard_similarity(tool_input, past_input) >= self.fuzzy_threshold
        )
        if fuzzy_matches >= self.exact_threshold:
            self.tool_history.append(current)
            return LoopDetectionResult(
                True, "fuzzy",
                f"Fuzzy loop detected: '{tool_name}' repeated with very similar arguments.", 0.85,
            )

        self.tool_history.append(current)
        return LoopDetectionResult(False, "none", "", 0.0)

    def check_output_stagnation(self, output: str) -> LoopDetectionResult:
        self.output_history.append(output)
        if len(self.output_history) < self.stagnation_window:
            return LoopDetectionResult(False, "none", "", 0.0)
        recent = self.output_history[-self.stagnation_window:]
        similarities = [
            self._jaccard_similarity(recent[i], recent[j])
            for i in range(len(recent)) for j in range(i + 1, len(recent))
        ]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        if avg_similarity >= self.fuzzy_threshold:
            return LoopDetectionResult(
                True, "stagnation",
                f"Output stagnation detected: the last {self.stagnation_window} outputs are too similar.",
                avg_similarity,
            )
        return LoopDetectionResult(False, "none", "", 0.0)

    def reset(self):
        self.tool_history.clear()
        self.output_history.clear()


loop_detector = LoopDetector()
MAX_RETRIES = 2
MAX_SEARCH_QUERIES_PER_ROUND = 3

# ============================================================
# SOURCE CELL 6
# ============================================================

# Web search + webpage reader
from bs4 import BeautifulSoup
import requests


def validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        try:
            ip_address = socket.gethostbyname(hostname)
        except socket.gaierror:
            return False
        parts = ip_address.split(".")
        if parts[0] == "10": return False
        if parts[0] == "192" and parts[1] == "168": return False
        if parts[0] == "172" and 16 <= int(parts[1]) <= 31: return False
        if parts[0] == "127": return False
        if ip_address == "0.0.0.0": return False
        return True
    except Exception:
        return False


def _format_tavily(response: dict, max_results: int) -> str:
    rows = []
    for item in (response.get("results") or [])[:max_results]:
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content", "")
        if url:
            rows.append(
                f"{len(rows)+1}. {title}\n"
                f"   Link: {url}\n"
                f"   Snippet: {content}"
            )
    return "\n\n".join(rows) if rows else "No results found."


def _duckduckgo_search(query: str, max_results: int = 5) -> str:
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.post(url, data={"q": query}, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"Search failed: {e}"

    soup = BeautifulSoup(response.text, "html.parser")
    lines = []
    for result in soup.find_all("div", class_="result", limit=max_results):
        title_tag = result.find("a", class_="result__a")
        snippet_tag = result.find("a", class_="result__snippet") or result.find(class_="result__snippet")
        if title_tag and snippet_tag:
            link = title_tag.get("href", "")
            if validate_url(link):
                lines.append(
                    f"{len(lines)+1}. {title_tag.get_text(strip=True)}\n"
                    f"   Link: {link}\n"
                    f"   Snippet: {snippet_tag.get_text(strip=True)}"
                )
    return "\n\n".join(lines) if lines else f"No results found for '{query}'."


@tool("search_web")
def search_web(query: str, max_results: int = 5) -> str:
    """Search the live web for products, prices, specifications, and Saudi availability."""
    # LoopDetector is applied inside the actual tool so every call from the Research
    # Agent is covered, not only calls made by the outer LangGraph node.
    check = loop_detector.check_tool_call("search_web", query)
    if check.is_looping:
        return f"LOOP BLOCKED: {check.message}"

    if tavily_client is not None:
        try:
            response = tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
            )
            result = _format_tavily(response, max_results)
            if result and "No results found" not in result:
                return result
        except Exception as e:
            logger.warning("Tavily search failed, falling back to DuckDuckGo: %s", e)

    return _duckduckgo_search(query, max_results)


@tool("read_webpage")
def read_webpage(url: str) -> str:
    """Read a public webpage and return cleaned text content."""
    if not validate_url(url):
        return "Error: Invalid or restricted URL."
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned = "\n".join(chunk for chunk in chunks if chunk)
        return cleaned[:12000]
    except Exception as e:
        return f"Error reading {url}: {e}"


RESEARCH_TOOLS = [search_web, read_webpage]


# ============================================================
# SOURCE CELL 8
# ============================================================

class Product(BaseModel):
    name: str = "Unknown"
    store: Optional[str] = "Unknown"
    url: Optional[str] = ""
    price_sar: Optional[float] = None
    cpu: Optional[str] = "Unknown"
    gpu: Optional[str] = "Unknown"
    ram_gb: Optional[int] = None
    storage_gb: Optional[int] = None
    display: Optional[str] = "Unknown"
    availability: Optional[str] = "Unknown"
    evidence: Optional[str] = ""


class ProductList(BaseModel):
    products: list[Product] = Field(default_factory=list)


class Requirements(BaseModel):
    country: str = "Saudi Arabia"
    currency: str = "SAR"
    category: str = "laptop"
    budget_max_sar: Optional[float] = None
    budget_min_sar: Optional[float] = None
    experience_level: Optional[str] = None
    use_cases: list[str] = Field(default_factory=list)
    priorities: list[str] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)


# ============================================================
# SOURCE CELL 9
# ============================================================

PLANNER_PROMPT = """
You are the Planning Agent for a Saudi product-decision system.
Your job is to turn a user's purchase request into a compact research plan.
Return JSON only with this exact shape:
{
  "search_queries": ["...", "...", "..."],
  "focus": ["..."],
  "verification_points": ["..."]
}

Rules:
- Default to Saudi Arabia and SAR unless the user explicitly specifies another market.
- Generate at most 3 highly targeted web-search queries.
- Include the relevant product category, budget, use case, and Saudi/SAR context.
- Prefer queries likely to surface Saudi retailers or official Saudi product pages.
- Search for products and actual configurations, not generic buying guides.
"""

REQUIREMENTS_PROMPT = """
You are the Requirements Agent.
Extract the user's purchasing needs into JSON only:
{
  "country": "Saudi Arabia",
  "currency": "SAR",
  "category": "laptop",
  "budget_max_sar": 5000,
  "budget_min_sar": null,
  "experience_level": null,
  "use_cases": [],
  "priorities": [],
  "hard_constraints": []
}

Do not invent a budget or requirement the user did not state. If the user does not state an experience level, use null.
For a laptop request, translate phrases such as beginner in AI into practical priorities such as NVIDIA GPU/CUDA, RAM, CPU, upgradeability, and price — but keep the final list concise.
"""

RESEARCHER_PROMPT = """
You are the Research Agent in a live product-decision system focused on Saudi Arabia.

You receive the user's requirements and preliminary web search results that were
already collected by the outer research node.

Analyze ONLY the supplied evidence.
Do NOT call tools.
Do NOT browse again.
Do NOT use general knowledge to fill missing facts.

Prioritize exact product configurations, Saudi availability, SAR prices, direct
product pages, and explicit CPU/GPU/RAM/storage evidence.

Do not treat a "starting from" price as proof of a higher specification.
If a field is missing, write Unknown.
If evidence conflicts, explicitly flag the conflict.

Return concise research notes with:
1) candidate products,
2) exact evidence and URLs,
3) price/specification ambiguity,
4) source-quality concerns.
"""

NORMALIZER_PROMPT = """
You are the Product Normalizer Agent.
Convert the research notes into JSON only, using this exact shape:
{
  "products": [
    {
      "name": "...",
      "store": "...",
      "url": "...",
      "price_sar": null,
      "cpu": "...",
      "gpu": "...",
      "ram_gb": null,
      "storage_gb": null,
      "display": "...",
      "availability": "...",
      "evidence": "..."
    }
  ]
}

Rules:
- Keep only products for which there is enough evidence to identify the configuration.
- Never infer missing specifications.
- price_sar must be the actual stated price for the cited configuration when available; otherwise null.
- Deduplicate the same product/configuration across repeated search results.
"""

COMPARISON_PROMPT = """
You are the Product Comparison Agent.
Given the user's requirements and normalized products, rank the products for THAT USER.
Do not reward a product for specs that are unknown.
Penalize products that exceed the budget or lack Saudi availability/evidence.

Return JSON only:
{
  "best_choice": "Product name",
  "confidence": 0.0,
  "ranked": [
    {"name": "...", "score": 91, "reason": "...", "tradeoffs": ["..."]}
  ]
}

The recommendation must balance use-case fit, GPU/CPU/RAM suitability, price, upgradeability, and portability according to the user's stated needs. Never treat missing price or missing critical specs as confirmed facts.
"""

CRITIC_PROMPT = """
You are the Quality Critic Agent.
Check the comparison before the system makes a final recommendation.

Return JSON only:
{
  "status": "PASS" or "FAIL",
  "confidence": 0.0,
  "issues": [],
  "recheck_queries": []
}

Fail when:
- the top choice is above the hard budget,
- the quoted price is not clearly tied to the configuration,
- the product is not reasonably supported as available in Saudi Arabia,
- the recommendation depends on missing critical specs,
- or the evidence conflicts materially.

If there are no products, return FAIL. If a hard budget is present and the top candidate has no verified price, return FAIL. If the research is strong, return PASS.
If FAIL, create targeted recheck_queries that would resolve the specific issue.
"""

FINAL_PROMPT = """
You are the Final Decision Agent.
Create a concise buying recommendation for the user.
Use only the supplied researched facts. Do not invent prices, specs, store names, URLs, or product names. Never mention a retailer unless it appears in the supplied product evidence. If products is empty, state that no fully verified match was found and do not add generic model recommendations.

Format:
- Best choice
- Price and Saudi store/source
- Why it fits the user's use case
- Key trade-offs
- 1-2 alternatives
- Confidence
- A one-line note that prices/availability can change

If the critic status is FAIL after the retry budget is exhausted, do NOT present an unverified product as a safe recommendation. Instead say that no fully verified match was found, summarize the best available evidence, and explain what needs manual verification.
"""

# ============================================================
# SOURCE CELL 10
# ============================================================

# Build the agents with create_agent (core LangChain requirement)
planner_agent = create_agent(model=llm, tools=[], system_prompt=PLANNER_PROMPT)
requirements_agent = create_agent(model=llm, tools=[], system_prompt=REQUIREMENTS_PROMPT)

# Research is already performed by research_node.
# This agent only analyzes the collected evidence to avoid nested tool loops.
research_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=RESEARCHER_PROMPT,
)

normalizer_agent = create_agent(model=llm, tools=[], system_prompt=NORMALIZER_PROMPT)
comparison_agent = create_agent(model=llm, tools=[], system_prompt=COMPARISON_PROMPT)
critic_agent = create_agent(model=llm, tools=[], system_prompt=CRITIC_PROMPT)
final_agent = create_agent(model=llm, tools=[], system_prompt=FINAL_PROMPT)

# ============================================================
# SOURCE CELL 11
# ============================================================

def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


@observe(name="agent_run", as_type="agent")
async def run_agent(agent, query: str, max_steps: int = 4) -> dict:
    result = await agent.ainvoke(
        {"messages": [("user", query)]},
        config={"recursion_limit": 2 * max_steps + 1},
    )
    messages = result["messages"]
    answer = content_to_text(messages[-1].content)
    total_tokens = sum(
        (getattr(m, "usage_metadata", None) or {}).get("total_tokens", 0)
        for m in messages
    )
    total_cost = sum(
        (getattr(m, "response_metadata", None) or {}).get("token_usage", {}).get("cost") or 0.0
        for m in messages
    )
    langfuse_context.update_current_observation(
        usage={"total_tokens": total_tokens},
        model=MODEL_NAME,
        metadata={"cost_usd": round(total_cost, 6)},
    )
    return {
        "answer": answer,
        "metadata": {
            "total_messages": len(messages),
            "total_tokens": total_tokens,
            "cost_usd": round(total_cost, 6),
        },
    }


def extract_json(text: str) -> Any:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_candidates = [i for i in [text.find("{"), text.find("[")] if i >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        for end_char in ["}", "]"]:
            end = text.rfind(end_char)
            if end > start:
                try:
                    return json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    pass
        raise


def merge_metrics(state: dict, stage_name: str, metadata: dict) -> dict:
    """Append every invocation instead of overwriting retries for the same stage."""
    metrics = {}
    for key, value in (state.get("metrics", {}) or {}).items():
        metrics[key] = list(value) if isinstance(value, list) else [value]
    metrics.setdefault(stage_name, []).append(dict(metadata))
    return metrics


def coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:[.,]\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


# ============================================================
# SOURCE CELL 12
# ============================================================

class ProductDecisionState(TypedDict, total=False):
    user_request: str
    requirements_json: str
    search_queries: list[str]
    used_queries: list[str]
    search_round: int
    raw_search_results: str
    research_notes: str
    products_json: str
    comparison_json: str
    critique_json: str
    retry_count: int
    loop_warnings: list[str]
    final_answer: str
    metrics: dict


# ============================================================
# SOURCE CELL 13
# ============================================================

async def planner_node(state: ProductDecisionState) -> dict:
    prompt = f"""
USER REQUEST:
{state['user_request']}

REQUIREMENTS:
{state.get('requirements_json', '{}')}

Create a compact research plan with no more than 3 search queries.
"""
    result = await run_agent(planner_agent, prompt)
    plan = extract_json(result["answer"])
    used = {q.strip().lower() for q in state.get("used_queries", [])}
    queries = [
        q.strip() for q in plan.get("search_queries", [])
        if isinstance(q, str) and q.strip() and q.strip().lower() not in used
    ]
    return {
        "search_queries": queries[:MAX_SEARCH_QUERIES_PER_ROUND],
        "metrics": merge_metrics(state, "planner", result["metadata"]),
    }


async def requirements_node(state: ProductDecisionState) -> dict:
    result = await run_agent(requirements_agent, f"USER REQUEST:\n{state['user_request']}")
    raw = extract_json(result["answer"])
    if raw.get("budget_max_sar") is not None:
        raw["budget_max_sar"] = coerce_number(raw["budget_max_sar"])
    if raw.get("budget_min_sar") is not None:
        raw["budget_min_sar"] = coerce_number(raw["budget_min_sar"])
    # Missing optional fields must remain None rather than causing a validation crash.
    if raw.get("experience_level") is None:
        raw["experience_level"] = None
    req = Requirements.model_validate(raw)
    return {
        "requirements_json": req.model_dump_json(),
        "metrics": merge_metrics(state, "requirements", result["metadata"]),
    }


async def research_node(state: ProductDecisionState) -> dict:
    queries = [
        q.strip() for q in state.get("search_queries", [])
        if isinstance(q, str) and q.strip()
    ]
    used_queries = list(state.get("used_queries", []))
    used_lower = {q.lower() for q in used_queries}
    queries = [q for q in queries if q.lower() not in used_lower][:MAX_SEARCH_QUERIES_PER_ROUND]

    if not queries:
        queries = ["best laptop Saudi Arabia SAR"]

    warnings = list(state.get("loop_warnings", []))

    async def safe_search(query: str):
        result = await asyncio.to_thread(search_web.invoke, {"query": query, "max_results": 5})
        if isinstance(result, str) and result.startswith("LOOP BLOCKED"):
            warnings.append(result)
        return result

    results = await asyncio.gather(*(safe_search(q) for q in queries))
    bundle = "\n\n===== SEARCH QUERY =====\n".join(
        f"{q}\n\n{res}" for q, res in zip(queries, results)
    )

    for q in queries:
        if q.lower() not in {x.lower() for x in used_queries}:
            used_queries.append(q)

    req = state.get("requirements_json", "{}")
    research_prompt = f"""
USER REQUEST:
{state['user_request']}

REQUIREMENTS JSON:
{req}

PRELIMINARY SEARCH RESULTS:
{bundle}

Previous critic feedback (if any):
{state.get('critique_json', '{}')}

Verify the strongest candidates. Use read_webpage for direct product pages where useful.
Do not invent any fact that is not present in the provided results or page content.
"""
    result = await run_agent(research_agent, research_prompt, max_steps=2)
    return {
        "raw_search_results": bundle,
        "research_notes": result["answer"],
        "loop_warnings": warnings,
        "used_queries": used_queries,
        "search_round": state.get("search_round", 0) + 1,
        "metrics": merge_metrics(state, "research", result["metadata"]),
    }


async def normalize_node(state: ProductDecisionState) -> dict:
    prompt = f"""
RESEARCH NOTES:
{state['research_notes']}

RAW SEARCH RESULTS:
{state.get('raw_search_results', '')}

Extract only products that are actually supported by the evidence.
"""
    result = await run_agent(normalizer_agent, prompt)
    data = extract_json(result["answer"])

    if not isinstance(data, dict):
        data = {"products": []}

    text_fields = [
        "name", "store", "url", "cpu", "gpu",
        "display", "availability", "evidence"
    ]

    clean_products = []
    for raw_product in data.get("products", []) or []:
        if not isinstance(raw_product, dict):
            continue

        p = dict(raw_product)

        # Prevent null / non-string optional fields from breaking Pydantic.
        for field_name in text_fields:
            value = p.get(field_name)
            if value is None or not isinstance(value, str) or not value.strip():
                p[field_name] = "" if field_name == "evidence" else "Unknown"
            else:
                p[field_name] = value.strip()

        p["price_sar"] = coerce_number(p.get("price_sar"))

        ram = coerce_number(p.get("ram_gb"))
        storage = coerce_number(p.get("storage_gb"))
        p["ram_gb"] = int(ram) if ram is not None else None
        p["storage_gb"] = int(storage) if storage is not None else None

        clean_products.append(p)

    data = {"products": clean_products}

    # Defensive validation: malformed model output should not crash the app.
    try:
        products = ProductList.model_validate(data)
    except ValidationError:
        safe_products = []
        for p in clean_products:
            try:
                safe_products.append(Product.model_validate(p))
            except ValidationError:
                continue
        products = ProductList(products=safe_products)
    updates = {
        "products_json": products.model_dump_json(),
        "metrics": merge_metrics(state, "normalizer", result["metadata"]),
    }
    if not products.products:
        updates["comparison_json"] = json.dumps({"best_choice": None, "confidence": 0.0, "ranked": []})
        updates["critique_json"] = json.dumps({
            "status": "FAIL",
            "confidence": 1.0,
            "issues": ["No verified product candidates were extracted from the available evidence."],
            "recheck_queries": ["Search for current Saudi retailer listings with exact configurations and prices."],
        })
    return updates


async def comparison_node(state: ProductDecisionState) -> dict:
    prompt = f"""
USER REQUEST:
{state['user_request']}

REQUIREMENTS:
{state['requirements_json']}

PRODUCTS:
{state['products_json']}

Compare the available products and rank them for this specific user.
Do not fill missing fields from general knowledge.
"""
    result = await run_agent(comparison_agent, prompt)
    comparison = extract_json(result["answer"])
    return {
        "comparison_json": json.dumps(comparison, ensure_ascii=False),
        "metrics": merge_metrics(state, "comparison", result["metadata"]),
    }


async def critic_node(state: ProductDecisionState) -> dict:
    prompt = f"""
USER REQUEST:
{state['user_request']}

REQUIREMENTS:
{state['requirements_json']}

PRODUCTS:
{state['products_json']}

COMPARISON:
{state['comparison_json']}

Check the recommendation strictly.
"""
    result = await run_agent(critic_agent, prompt)
    critique = extract_json(result["answer"])
    critique_text = json.dumps(critique, ensure_ascii=False)
    stagnation = loop_detector.check_output_stagnation(critique_text)

    warnings = list(state.get("loop_warnings", []))
    if stagnation.is_looping:
        warnings.append(stagnation.message)
        critique["status"] = "FAIL"
        critique.setdefault("issues", []).append("Stagnation detected; force a fresh research path.")
        critique.setdefault("recheck_queries", []).append("Find a different Saudi retail source for the top laptop candidate.")

    return {
        "critique_json": json.dumps(critique, ensure_ascii=False),
        "loop_warnings": warnings,
        "metrics": merge_metrics(state, "critic", result["metadata"]),
    }


async def retry_planner_node(state: ProductDecisionState) -> dict:
    prompt = f"""
The previous recommendation failed quality control.

USER REQUEST:
{state['user_request']}
REQUIREMENTS:
{state['requirements_json']}
PREVIOUS CRITIQUE:
{state['critique_json']}
PREVIOUSLY USED QUERIES:
{json.dumps(state.get('used_queries', []), ensure_ascii=False)}

Generate up to 3 NEW web-search queries that target the specific evidence gaps.
Do not repeat previously used queries.
Return JSON only:
{{"search_queries": ["..."]}}
"""
    result = await run_agent(planner_agent, prompt)
    plan = extract_json(result["answer"])
    used = {q.lower() for q in state.get("used_queries", [])}
    new_queries = [
        q.strip() for q in plan.get("search_queries", [])
        if isinstance(q, str) and q.strip() and q.strip().lower() not in used
    ]
    return {
        "search_queries": new_queries[:MAX_SEARCH_QUERIES_PER_ROUND],
        "retry_count": state.get("retry_count", 0) + 1,
        "metrics": merge_metrics(state, "retry_planner", result["metadata"]),
    }


async def final_node(state: ProductDecisionState) -> dict:
    products = extract_json(state.get("products_json", "{}"))
    product_items = products.get("products", []) if isinstance(products, dict) else []
    if not product_items:
        # Deterministic safety fallback: no model-generated facts when evidence is empty.
        return {
            "final_answer": (
                "**No fully verified match found**\n\n"
                "No product listing met the evidence requirements in the available Saudi web research. "
                "The system will not invent a product, price, specification, store, or URL.\n\n"
                "The next step is to re-run the search or provide product links for direct comparison."
            ),
            "metrics": merge_metrics(state, "final", {"total_messages": 0, "total_tokens": 0, "cost_usd": 0.0}),
        }

    prompt = f"""
USER REQUEST:
{state['user_request']}

REQUIREMENTS:
{state['requirements_json']}

PRODUCTS:
{state['products_json']}

COMPARISON:
{state['comparison_json']}

CRITIC:
{state['critique_json']}

RESEARCH NOTES:
{state['research_notes']}

Create the final buying recommendation.
Use ONLY the supplied product evidence. Do not invent prices, specs, store names, URLs, or products.
If critic status is FAIL after the retry budget, clearly label the recommendation as unverified instead of presenting it as a safe purchase.
"""
    result = await run_agent(final_agent, prompt)
    return {
        "final_answer": result["answer"],
        "metrics": merge_metrics(state, "final", result["metadata"]),
    }


# ============================================================
# SOURCE CELL 14
# ============================================================

def route_after_normalize(state: ProductDecisionState):
    products = extract_json(state.get("products_json", "{}"))
    if not products.get("products"):
        return "no_products"
    return "compare"


def route_after_critic(state: ProductDecisionState):
    critique = extract_json(state.get("critique_json", "{}"))
    status = str(critique.get("status", "FAIL")).upper()
    retries = state.get("retry_count", 0)

    if status == "PASS":
        return "final"
    if retries < MAX_RETRIES:
        return "retry"
    return "final"


# ============================================================
# GRAPH BUILD
# ============================================================

graph = StateGraph(ProductDecisionState)

graph.add_node("requirements", requirements_node)
graph.add_node("planner", planner_node)
graph.add_node("research", research_node)
graph.add_node("normalize", normalize_node)
graph.add_node("compare", comparison_node)
graph.add_node("critic", critic_node)
graph.add_node("retry_planner", retry_planner_node)
graph.add_node("final", final_node)

graph.add_edge(START, "requirements")
graph.add_edge("requirements", "planner")
graph.add_edge("planner", "research")
graph.add_edge("research", "normalize")

graph.add_conditional_edges(
    "normalize",
    route_after_normalize,
    {"compare": "compare", "no_products": "final"},
)

graph.add_edge("compare", "critic")
graph.add_conditional_edges(
    "critic",
    route_after_critic,
    {"final": "final", "retry": "retry_planner"},
)
graph.add_edge("retry_planner", "research")
graph.add_edge("final", END)

memory = InMemorySaver()
pipeline = graph.compile(checkpointer=memory)

# Graph compiled successfully.


# ============================================================
# PIPELINE RUNNER
# ============================================================

async def run_pipeline(query: str, thread_id: str) -> dict:
    # Fresh reliability state for each independent run
    loop_detector.reset()

    initial_state = {
        "user_request": query,
        "retry_count": 0,
        "search_round": 0,
        "used_queries": [],
        "loop_warnings": [],
        "metrics": {},
    }

    result = await pipeline.ainvoke(
        initial_state,
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 30,
        },
    )

    return result


def run_pipeline_sync(query: str, thread_id: str) -> dict:
    """Synchronous wrapper for Streamlit / standard Python execution."""
    return asyncio.run(run_pipeline(query, thread_id))
