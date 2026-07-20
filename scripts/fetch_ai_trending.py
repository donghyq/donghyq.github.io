#!/usr/bin/env python3
"""
Fetch AI-related trending repos from GitHub and classify them into categories.
Outputs a JSON file consumed by the About page's AI Trending section.

Usage:
  python scripts/fetch_ai_trending.py [--out assets/data/ai-trending.json]

Environment variables:
  GITHUB_TOKEN  - optional, raises rate limit from 10 to 30 req/min
  DEEPSEEK_API_KEY - optional, enables LLM-powered Chinese summaries
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

# Create an SSL context that doesn't verify certificates (for corporate proxy environments)
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

# ── Configuration ──────────────────────────────────────────────────────────

SEARCH_QUERIES = [
    # (keyword, per_page)
    ("llm",                 6),
    ("large language model", 4),
    ("RAG retrieval",       4),
    ("vector search",       4),
    ("AI agent",            6),
    ("transformer model",   4),
    ("AI coding",           4),
    ("LLM inference",       4),
    ("MCP server",          4),
]

DAYS_LOOKBACK = 30
MIN_STARS = 30

# Categories with matching keywords (checked against repo name + description + topics)
CATEGORIES = [
    {
        "id": "agent-tools",
        "name": "🧩 Agent 技能与工具",
        "desc": "把工程、产品、设计能力封装成可复用的 Skill / 工具模块",
        "keywords": ["agent", "skill", "tool", "mcp", "plugin", "extension",
                     "workflow", "automation", "cli", "sdk"],
    },
    {
        "id": "info-search",
        "name": "🔍 信息搜索与抓取",
        "desc": "主动抓取社区、社媒、网页、GitHub 等外部信号",
        "keywords": ["search", "scrape", "crawl", "spider", "fetch",
                     "browse", "retrieval", "rag", "index", "web"],
    },
    {
        "id": "context-infra",
        "name": "🧠 上下文与推理基础设施",
        "desc": "压缩、转换、整理输入，降低 Token 成本，加速推理",
        "keywords": ["context", "compress", "token", "inference", "serving",
                     "kv-cache", "vllm", "sglang", "quantiz", "memory",
                     "distill", "pruning", "gguf", "ggml", "ollama"],
    },
    {
        "id": "model-training",
        "name": "🔬 模型训练与微调",
        "desc": "预训练、微调、对齐、评估等模型构建环节",
        "keywords": ["train", "fine-tune", "finetune", "lora", "rlhf",
                     "alignment", "pretrain", "eval", "benchmark", "dataset"],
    },
    {
        "id": "app-frontend",
        "name": "🎨 应用与前端",
        "desc": "面向终端用户的 AI 应用、UI 组件、聊天界面",
        "keywords": ["chat", "ui", "app", "frontend", "interface", "demo",
                     "playground", "assistant", "copilot", "client", "gui"],
    },
]

DEFAULT_CATEGORY = {
    "id": "other",
    "name": "📦 其他热门",
    "desc": "值得关注的其他 AI/ML 开源项目",
}

# ── Helpers ────────────────────────────────────────────────────────────────

def gh_request(url: str, token: str | None = None) -> dict:
    """Make a GitHub API request with optional auth."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  ⚠ HTTP {e.code} for {url}", file=sys.stderr)
        return {}


def classify_repo(repo: dict) -> str:
    """Return category id based on repo metadata."""
    text = " ".join([
        (repo.get("name") or ""),
        (repo.get("description") or ""),
        " ".join(repo.get("topics") or []),
    ]).lower()

    scores = {}
    for cat in CATEGORIES:
        score = sum(1 for kw in cat["keywords"] if kw in text)
        if score > 0:
            scores[cat["id"]] = score

    if scores:
        return max(scores, key=scores.get)
    return DEFAULT_CATEGORY["id"]


def generate_summary(repo: dict) -> str:
    """Generate a one-line Chinese summary for a repo.
    Uses DeepSeek API if available, otherwise falls back to rule-based."""

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if api_key:
        return _llm_summary(repo, api_key)
    return _rule_summary(repo)


def _llm_summary(repo: dict, api_key: str) -> str:
    """Call DeepSeek API to generate Chinese summary."""
    desc = (repo.get("description") or "")[:200]
    name = repo.get("full_name", "")
    lang = repo.get("language") or "未知"
    topics = ", ".join((repo.get("topics") or [])[:5])
    stars = repo.get("stargazers_count", 0)

    prompt = (
        f"请用一句简洁的中文（30字以内）概括这个 GitHub 项目的核心功能：\n"
        f"项目：{name}\n"
        f"描述：{desc}\n"
        f"语言：{lang}，标签：{topics}，Stars：{stars}\n"
        f"只输出摘要，不要加引号或前缀。"
    )

    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20, context=_ssl_ctx) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  ⚠ LLM summary failed for {name}: {e}", file=sys.stderr)
        return _rule_summary(repo)


def _rule_summary(repo: dict) -> str:
    """Fallback: build a Chinese summary from repo metadata."""
    desc = (repo.get("description") or "").strip()
    lang = repo.get("language") or ""
    name = repo.get("name", "")

    # Try to keep it short and informative
    if not desc:
        if lang:
            return f"基于 {lang} 的 {name} 项目"
        return f"{name} 开源项目"

    # If description is already short enough, use it directly
    if len(desc) <= 80:
        return desc

    # Truncate at sentence boundary
    for sep in [". ", "。", "，", ", ", " - ", " — "]:
        idx = desc.find(sep)
        if 10 < idx < 80:
            return desc[:idx + (1 if sep.endswith(" ") else len(sep))].strip()

    return desc[:77] + "..."


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch AI trending repos")
    parser.add_argument("--out", default="assets/data/ai-trending.json",
                        help="Output JSON path")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS_LOOKBACK)).strftime("%Y-%m-%d")

    print(f"🔍 Fetching AI trending repos (since {since}, min ⭐ {MIN_STARS})")

    seen_ids = set()
    all_repos = []

    for kw, per_page in SEARCH_QUERIES:
        q = urllib.request.quote(f"{kw} created:>{since} stars:>{MIN_STARS}")
        url = f"https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page={per_page}"
        print(f"  → Searching: {kw}")
        data = gh_request(url, token)
        items = data.get("items", [])
        for repo in items:
            rid = repo["id"]
            if rid not in seen_ids:
                seen_ids.add(rid)
                all_repos.append(repo)
        time.sleep(1)  # Rate limit friendly

    print(f"📦 Total unique repos: {len(all_repos)}")

    # Classify and summarize
    categorized = {}
    for repo in all_repos:
        cat_id = classify_repo(repo)
        if cat_id not in categorized:
            categorized[cat_id] = []

        summary = generate_summary(repo)
        print(f"  ✓ {repo['full_name']} → {cat_id}: {summary[:40]}")

        categorized[cat_id].append({
            "name": repo["full_name"],
            "url": repo["html_url"],
            "description": (repo.get("description") or "")[:150],
            "summary_zh": summary,
            "stars": repo["stargazers_count"],
            "forks": repo["forks_count"],
            "language": repo.get("language") or "",
            "topics": (repo.get("topics") or [])[:6],
            "created_at": repo.get("created_at", ""),
        })

    # Sort each category by stars descending, keep top N
    MAX_PER_CAT = 4
    for cat_id in categorized:
        categorized[cat_id].sort(key=lambda r: r["stars"], reverse=True)
        categorized[cat_id] = categorized[cat_id][:MAX_PER_CAT]

    # Build output with ordered categories
    cat_lookup = {c["id"]: c for c in CATEGORIES}
    cat_lookup[DEFAULT_CATEGORY["id"]] = DEFAULT_CATEGORY

    output_categories = []
    for cat in [*CATEGORIES, DEFAULT_CATEGORY]:
        if cat["id"] in categorized:
            output_categories.append({
                "id": cat["id"],
                "name": cat["name"],
                "desc": cat["desc"],
                "repos": categorized[cat["id"]],
            })

    # Summary paragraph
    total = sum(len(c["repos"]) for c in output_categories)
    top_langs = {}
    for c in output_categories:
        for r in c["repos"]:
            if r["language"]:
                top_langs[r["language"]] = top_langs.get(r["language"], 0) + 1
    top3_langs = sorted(top_langs, key=top_langs.get, reverse=True)[:3]

    digest = (
        f"过去 {DAYS_LOOKBACK} 天，GitHub 上共涌现 {total} 个值得关注的 AI 开源项目，"
        f"主力语言为 {', '.join(top3_langs)}。"
    )
    if output_categories:
        top_cat = max(output_categories, key=lambda c: sum(r["stars"] for r in c["repos"]))
        digest += f"最热门方向是「{top_cat['name'].split(' ', 1)[1]}」。"

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
        "lookback_days": DAYS_LOOKBACK,
        "digest": digest,
        "categories": output_categories,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Written to {args.out}")
    print(f"   {len(output_categories)} categories, {total} repos")
    print(f"   Digest: {digest}")


if __name__ == "__main__":
    main()
