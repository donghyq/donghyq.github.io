#!/usr/bin/env python3
"""
Fetch AI-related trending data from multiple sources:
  1. GitHub Search API — new repos by keyword
  2. GitHub Trending — daily/weekly trending page
  3. ArXiv API — latest AI/ML papers

Outputs a JSON file consumed by the About page's AI Trending section.

Usage:
  python scripts/fetch_ai_trending.py [--out assets/data/ai-trending.json]

Environment variables:
  GITHUB_TOKEN     - optional, raises rate limit from 10→30 req/min
  DEEPSEEK_API_KEY - optional, enables LLM-powered Chinese summaries
"""

import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from html import unescape

# ── SSL (corporate proxy workaround) ──────────────────────────────────────
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

# ── Configuration ─────────────────────────────────────────────────────────

SEARCH_QUERIES = [
    ("llm",                  6),
    ("large language model", 4),
    ("RAG retrieval",        4),
    ("vector search",        4),
    ("AI agent",             6),
    ("transformer model",    4),
    ("AI coding",            4),
    ("LLM inference",        4),
    ("MCP server",           4),
]

DAYS_LOOKBACK = 30
MIN_STARS = 30

# GitHub Trending scrape config
TRENDING_LANGUAGES = ["", "python", "typescript", "rust"]  # "" = all languages
TRENDING_SINCE = "weekly"  # daily / weekly / monthly

# ArXiv config
ARXIV_QUERIES = [
    "cat:cs.CL",   # Computation and Language (NLP/LLM)
    "cat:cs.AI",   # Artificial Intelligence
    "cat:cs.LG",   # Machine Learning
    "cat:cs.IR",   # Information Retrieval
]
ARXIV_MAX_RESULTS = 10  # per query
ARXIV_DAYS = 7          # look back N days

# Categories for GitHub repos
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

# ── HTTP helper ───────────────────────────────────────────────────────────

def _http_get(url: str, headers: dict | None = None, timeout: int = 15) -> bytes:
    """Raw HTTP GET, returns bytes. Raises on error."""
    hdrs = headers or {}
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
        return resp.read()


def gh_request(url: str, token: str | None = None) -> dict:
    """GitHub API JSON request."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        return json.loads(_http_get(url, headers).decode())
    except urllib.error.HTTPError as e:
        print(f"  ⚠ HTTP {e.code} for {url}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"  ⚠ Request failed: {e}", file=sys.stderr)
        return {}


# ── Classification ────────────────────────────────────────────────────────

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


# ── Summaries ─────────────────────────────────────────────────────────────

def generate_summary(item: dict, item_type: str = "repo") -> str:
    """Generate Chinese summary. Uses DeepSeek if available."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if api_key:
        result = _llm_summary(item, api_key, item_type)
        return result
    else:
        print("  ℹ No DEEPSEEK_API_KEY, using rule-based summary", file=sys.stderr)
    return _rule_summary(item, item_type)


def _llm_summary(item: dict, api_key: str, item_type: str) -> str:
    """Call DeepSeek API for Chinese summary."""
    if item_type == "paper":
        title = item.get("title", "")
        abstract = item.get("abstract", "")[:300]
        prompt = (
            f"请用一句简洁的中文（30-50字）概括这篇 AI 论文的核心贡献和亮点。\n"
            f"标题：{title}\n"
            f"摘要：{abstract}\n"
            f"要求：纯中文输出，突出方法创新点和关键结果数据（如有），"
            f"不要加引号、句号或任何前缀。\n"
            f"示例风格：提出基于注意力蒸馏的长文本压缩方法，将上下文窗口扩展至 128K 且推理速度提升 3 倍"
        )
    else:
        desc = (item.get("description") or "")[:200]
        name = item.get("full_name") or item.get("name", "")
        lang = item.get("language") or "未知"
        topics = ", ".join((item.get("topics") or [])[:5])
        stars = item.get("stargazers_count") or item.get("stars", 0)
        prompt = (
            f"请用一句简洁的中文（20-40字）介绍这个 GitHub 开源项目的核心功能和亮点。\n"
            f"项目：{name}\n描述：{desc}\n"
            f"语言：{lang}，标签：{topics}，Stars：{stars}\n"
            f"要求：纯中文输出，像科技博主写的一句话点评，突出这个项目解决什么问题、有什么独特之处。"
            f"不要加引号、句号或任何前缀。不要以「这是」「该项目」开头。\n"
            f"示例风格：\n"
            f"- 面向 AI 编码 Agent 的生产级工程技能库，让 AI 更像成熟工程师\n"
            f"- AI Agent 上下文压缩工具，可减少 60-95% Token 消耗同时保持答案质量\n"
            f"- 微软开源的文件转 Markdown 工具，适合把 Office、PDF 等资料转成 RAG 可读格式"
        )

    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=_ssl_ctx) as resp:
            result = json.loads(resp.read().decode())
            summary = result["choices"][0]["message"]["content"].strip()
            if summary:
                return summary
    except Exception as e:
        print(f"  ⚠ LLM summary failed: {e}", file=sys.stderr)
    return _rule_summary(item, item_type)


def _rule_summary(item: dict, item_type: str = "repo") -> str:
    """Fallback rule-based summary.
    
    When no LLM API is available, generate a brief Chinese-wrapped summary
    using keyword-based templates to produce natural Chinese descriptions.
    """
    if item_type == "paper":
        title = item.get("title", "")
        return title[:80] if len(title) <= 80 else title[:77] + "..."

    desc = (item.get("description") or "").strip()
    name = item.get("name") or item.get("full_name", "")
    lang = item.get("language") or ""

    if not desc:
        return f"基于 {lang} 的 {name} 开源项目" if lang else f"{name} 开源项目"

    # Clean up emoji at the start
    clean_desc = re.sub(r'^[\U0001F300-\U0001FAFF\U00002702-\U000027B0\u2600-\u27FF\s]+', '', desc).strip()
    if not clean_desc:
        clean_desc = desc

    # If already Chinese, truncate and return
    if re.search(r'[\u4e00-\u9fff]', clean_desc):
        if len(clean_desc) <= 80:
            return clean_desc
        for sep in ["。", "，", "；", "、"]:
            idx = clean_desc.find(sep)
            if 5 < idx < 80:
                return clean_desc[:idx + len(sep)].strip()
        return clean_desc[:77] + "..."

    # ── Keyword-based Chinese template generation ──
    text = f"{name} {clean_desc}".lower()

    # Template patterns: (keywords_any_match, chinese_summary)
    # Pure Chinese descriptions, no English text mixed in
    _TEMPLATES = [
        # Agent / Skill / Tool
        (["coding agent", "code agent", "ai coding"],
         "AI 编程助手，在终端中运行的轻量级智能编码工具"),
        (["agent skill", "skill for", "agent-skill", "skills for"],
         "面向 AI Agent 的生产级技能模块库，让 AI 更像成熟工程师"),
        (["multi-agent", "multi agent"],
         "多 Agent 协作框架，支持多智能体协同完成复杂任务"),
        (["red team", "security", "offensive"],
         "AI 安全攻防平台，自动化红队测试与漏洞发现"),
        (["ai agent", "autonomous agent"],
         "AI Agent 开发框架，构建自主智能体应用"),
        (["mcp server", "mcp tool"],
         "MCP 协议工具服务，为 AI Agent 提供标准化工具接口"),
        (["workflow", "orchestrat"],
         "AI 工作流编排引擎，支持灵活的任务调度与自动化"),
        (["auth gateway", "connector", "oauth"],
         "开源认证网关，一键连接 1000+ SaaS 服务到 AI Agent"),

        # Search / RAG / Retrieval
        (["rag", "retrieval augmented", "retrieval-augmented", "knowledge base"],
         "检索增强生成（RAG）框架，构建本地知识库问答系统"),
        (["vector search", "vector database", "similarity search", "embedding"],
         "向量检索引擎，支持高性能相似度搜索"),
        (["web search", "search engine", "web scrape", "crawler"],
         "AI 驱动的网页搜索与数据抓取工具"),

        # LLM / Inference
        (["llm inference", "inference engine", "serving", "inference serv", "distributed inference"],
         "LLM 分布式推理框架，支持大规模模型高效部署"),
        (["kv cache", "kv-cache"],
         "LLM KV Cache 优化方案，加速大模型推理"),
        (["quantiz", "gguf", "ggml"],
         "模型量化压缩工具，降低部署成本同时保持模型质量"),
        (["context window", "context compress", "long context", "context7"],
         "上下文管理工具，为 AI 提供最新的代码文档与上下文"),
        (["fine-tun", "finetun", "lora"],
         "模型微调框架，支持低资源高效训练"),
        (["heterogeneous", "cpu.*gpu", "gpu.*cpu"],
         "异构计算推理框架，支持 CPU/GPU 混合部署大模型"),
        (["local.*llm", "llm.*local", "run.*local"],
         "本地 LLM 运行指南，涵盖本地大模型部署全流程"),
        (["llm", "large language model", "language model"],
         "大语言模型应用工具，简化 LLM 开发与集成"),

        # App / UI / Chat
        (["chat", "chatbot", "assistant", "copilot"],
         "AI 智能助手应用，支持本地部署的对话式 AI"),
        (["video editor", "video edit"],
         "AI 视频编辑工具，开源免费的智能剪辑方案"),
        (["gui", "desktop app", "immediate mode gui"],
         "轻量级即时模式 GUI 框架，适合快速构建桌面应用"),
        (["roadmap", "learning", "tutorial"],
         "开发者学习路线图，交互式技术成长指南"),
        (["ui", "frontend", "interface"],
         "AI 前端工具，构建智能交互界面"),

        # Infra
        (["javascript runtime", "runtime"],
         "高性能 JavaScript 运行时，极速的开发体验"),
        (["python package", "package manager", "linter", "formatter"],
         "高性能 Python 开发工具链，极速的包管理与代码格式化"),
        (["container", "docker", "kubernetes"],
         "容器化工具，轻量级虚拟化与部署方案"),
    ]

    for keywords, summary in _TEMPLATES:
        if any(kw in text for kw in keywords):
            return summary

    # Final fallback: generic Chinese description
    if lang:
        return f"基于 {lang} 的开源项目，{name.split('/')[-1] if '/' in name else name}"
    return f"开源项目 {name.split('/')[-1] if '/' in name else name}"


# ══════════════════════════════════════════════════════════════════════════
# Source 1: GitHub Search API
# ══════════════════════════════════════════════════════════════════════════

def fetch_github_search(token: str | None) -> list[dict]:
    """Fetch repos via GitHub Search API."""
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS_LOOKBACK)).strftime("%Y-%m-%d")
    print(f"🔍 [GitHub Search] Fetching repos (since {since}, min ⭐ {MIN_STARS})")

    seen_ids = set()
    results = []

    for kw, per_page in SEARCH_QUERIES:
        q = urllib.request.quote(f"{kw} created:>{since} stars:>{MIN_STARS}")
        url = f"https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page={per_page}"
        print(f"  → {kw}")
        data = gh_request(url, token)
        for repo in data.get("items", []):
            if repo["id"] not in seen_ids:
                seen_ids.add(repo["id"])
                results.append(repo)
        time.sleep(1)

    print(f"  📦 {len(results)} unique repos from Search API")
    return results


# ══════════════════════════════════════════════════════════════════════════
# Source 2: GitHub Trending (HTML scrape)
# ══════════════════════════════════════════════════════════════════════════

def fetch_github_trending() -> list[dict]:
    """Scrape GitHub Trending page for popular repos."""
    print(f"🔥 [GitHub Trending] Fetching {TRENDING_SINCE} trending repos")
    all_repos = []
    seen = set()

    for lang in TRENDING_LANGUAGES:
        lang_path = f"/{lang}" if lang else ""
        url = f"https://github.com/trending{lang_path}?since={TRENDING_SINCE}"
        print(f"  → {url}")
        try:
            html = _http_get(url, {
                "User-Agent": "Mozilla/5.0 (compatible; ai-trending-bot/1.0)"
            }).decode("utf-8", errors="replace")
            repos = _parse_trending_html(html)
            for r in repos:
                key = r["full_name"]
                if key not in seen:
                    seen.add(key)
                    all_repos.append(r)
        except Exception as e:
            print(f"  ⚠ Trending fetch failed for lang={lang}: {e}", file=sys.stderr)
        time.sleep(1)

    print(f"  📦 {len(all_repos)} unique repos from Trending")
    return all_repos


def _parse_trending_html(html: str) -> list[dict]:
    """Parse GitHub trending HTML to extract repo info."""
    repos = []
    # Each repo is in an <article class="Box-row">
    articles = re.findall(
        r'<article\s+class="Box-row">(.*?)</article>',
        html, re.DOTALL
    )

    for article in articles:
        # Repo name: <h2 ...><a ... href="/owner/name" ...>...</a></h2>
        # Must avoid matching stargazers/forks links — only match href with exactly one slash (owner/repo)
        name_match = re.search(r'<h2[^>]*>.*?<a\s[^>]*href="/([^"/]+/[^"/]+)"', article, re.DOTALL)
        if not name_match:
            continue
        full_name = name_match.group(1).strip()

        # Description
        desc_match = re.search(r'<p\s+class="[^"]*col-9[^"]*"[^>]*>(.*?)</p>', article, re.DOTALL)
        description = unescape(desc_match.group(1).strip()) if desc_match else ""
        description = re.sub(r'<[^>]+>', '', description).strip()

        # Language
        lang_match = re.search(r'<span\s+itemprop="programmingLanguage">(.*?)</span>', article)
        language = lang_match.group(1).strip() if lang_match else ""

        # Stars (total)
        stars_match = re.search(
            r'<a[^>]*href="/[^"]+/stargazers"[^>]*>\s*(?:<[^>]+>\s*)*\s*([\d,]+)\s*</a>',
            article, re.DOTALL
        )
        stars = int(stars_match.group(1).replace(",", "")) if stars_match else 0

        # Stars this period
        period_match = re.search(r'([\d,]+)\s+stars?\s+(?:today|this\s+week|this\s+month)', article)
        stars_period = int(period_match.group(1).replace(",", "")) if period_match else 0

        # Forks
        forks_match = re.search(
            r'<a[^>]*href="/[^"]+/forks"[^>]*>\s*(?:<[^>]+>\s*)*\s*([\d,]+)\s*</a>',
            article, re.DOTALL
        )
        forks = int(forks_match.group(1).replace(",", "")) if forks_match else 0

        repos.append({
            "full_name": full_name,
            "name": full_name.split("/")[-1] if "/" in full_name else full_name,
            "description": description,
            "language": language,
            "stargazers_count": stars,
            "stars_period": stars_period,
            "forks_count": forks,
            "html_url": f"https://github.com/{full_name}",
            "topics": [],
            "source": "trending",
        })

    return repos


# ══════════════════════════════════════════════════════════════════════════
# Source 3: ArXiv API
# ══════════════════════════════════════════════════════════════════════════

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

def fetch_arxiv_papers() -> list[dict]:
    """Fetch recent AI/ML papers from ArXiv API."""
    print(f"📄 [ArXiv] Fetching papers from last {ARXIV_DAYS} days")
    all_papers = []
    seen_ids = set()

    # Try multiple ArXiv endpoints (export.arxiv.org sometimes slow behind corp proxy)
    arxiv_hosts = ["https://export.arxiv.org", "http://export.arxiv.org"]

    for query in ARXIV_QUERIES:
        print(f"  → {query}")
        fetched = False
        for host in arxiv_hosts:
            url = (
                f"{host}/api/query?"
                f"search_query={urllib.request.quote(query)}"
                f"&sortBy=submittedDate&sortOrder=descending"
                f"&max_results={ARXIV_MAX_RESULTS}"
            )
            try:
                xml_data = _http_get(url, timeout=30).decode("utf-8")
                papers = _parse_arxiv_xml(xml_data)
                for p in papers:
                    if p["id"] not in seen_ids:
                        seen_ids.add(p["id"])
                        all_papers.append(p)
                fetched = True
                break
            except Exception as e:
                print(f"    ⚠ {host} failed: {e}", file=sys.stderr)
        if not fetched:
            print(f"    ✗ All ArXiv hosts failed for {query}", file=sys.stderr)
        time.sleep(1)

    # Filter by date
    cutoff = datetime.now(timezone.utc) - timedelta(days=ARXIV_DAYS)
    recent = [p for p in all_papers if p.get("published_dt") and p["published_dt"] >= cutoff]
    recent.sort(key=lambda p: p.get("published_dt", cutoff), reverse=True)

    print(f"  📦 {len(recent)} papers from last {ARXIV_DAYS} days (of {len(all_papers)} total)")
    return recent[:20]  # cap at 20


def _parse_arxiv_xml(xml_data: str) -> list[dict]:
    """Parse ArXiv Atom XML feed."""
    papers = []
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return papers

    for entry in root.findall("atom:entry", ARXIV_NS):
        arxiv_id_el = entry.find("atom:id", ARXIV_NS)
        title_el = entry.find("atom:title", ARXIV_NS)
        summary_el = entry.find("atom:summary", ARXIV_NS)
        published_el = entry.find("atom:published", ARXIV_NS)

        if arxiv_id_el is None or title_el is None:
            continue

        arxiv_id = (arxiv_id_el.text or "").strip()
        title = re.sub(r'\s+', ' ', (title_el.text or "").strip())
        abstract = re.sub(r'\s+', ' ', (summary_el.text or "").strip()) if summary_el is not None else ""
        published = (published_el.text or "").strip() if published_el is not None else ""

        # Parse date
        published_dt = None
        if published:
            try:
                published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Authors
        authors = []
        for author_el in entry.findall("atom:author", ARXIV_NS):
            name_el = author_el.find("atom:name", ARXIV_NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        # Categories
        categories = []
        for cat_el in entry.findall("{http://arxiv.org/schemas/atom}primary_category"):
            term = cat_el.get("term", "")
            if term:
                categories.append(term)
        for cat_el in entry.findall("atom:category", ARXIV_NS):
            term = cat_el.get("term", "")
            if term and term not in categories:
                categories.append(term)

        # PDF link
        pdf_url = ""
        for link_el in entry.findall("atom:link", ARXIV_NS):
            if link_el.get("title") == "pdf":
                pdf_url = link_el.get("href", "")
                break

        # Clean arxiv_id to get abs URL
        abs_url = arxiv_id  # already http://arxiv.org/abs/XXXX

        papers.append({
            "id": arxiv_id,
            "title": title,
            "abstract": abstract[:500],
            "authors": authors[:5],  # first 5 authors
            "categories": categories[:5],
            "published": published,
            "published_dt": published_dt,
            "url": abs_url,
            "pdf_url": pdf_url,
        })

    return papers


# ── ArXiv topic classification ────────────────────────────────────────────

PAPER_TOPICS = [
    {
        "id": "llm-reasoning",
        "name": "🧠 LLM 推理与优化",
        "keywords": ["reasoning", "chain-of-thought", "inference", "kv cache",
                     "speculative decoding", "quantiz", "pruning", "distill",
                     "efficient", "serving", "scaling", "long context"],
    },
    {
        "id": "rag-retrieval",
        "name": "🔍 RAG 与检索增强",
        "keywords": ["retrieval", "rag", "knowledge", "document", "embedding",
                     "vector", "dense retrieval", "passage", "open-domain"],
    },
    {
        "id": "agent-planning",
        "name": "🤖 Agent 与规划",
        "keywords": ["agent", "planning", "tool use", "function call",
                     "multi-agent", "autonomous", "self-refin", "reflection"],
    },
    {
        "id": "multimodal",
        "name": "🖼️ 多模态",
        "keywords": ["multimodal", "vision", "image", "video", "visual",
                     "vlm", "text-to-image", "diffusion", "generation"],
    },
    {
        "id": "alignment-safety",
        "name": "🛡️ 对齐与安全",
        "keywords": ["alignment", "rlhf", "dpo", "safety", "harmless",
                     "jailbreak", "red team", "preference", "reward model"],
    },
]

DEFAULT_PAPER_TOPIC = {"id": "other-paper", "name": "📝 其他前沿"}


def classify_paper(paper: dict) -> str:
    """Classify a paper into a topic."""
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    scores = {}
    for topic in PAPER_TOPICS:
        score = sum(1 for kw in topic["keywords"] if kw in text)
        if score > 0:
            scores[topic["id"]] = score
    if scores:
        return max(scores, key=scores.get)
    return DEFAULT_PAPER_TOPIC["id"]


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fetch AI trending data")
    parser.add_argument("--out", default="assets/data/ai-trending.json")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")

    # ── 1. GitHub Search API ──
    search_repos = fetch_github_search(token)

    # ── 2. GitHub Trending ──
    trending_repos = fetch_github_trending()

    # ── 3. Merge & deduplicate GitHub repos ──
    seen_names = set()
    all_repos = []
    # Trending repos take priority (they are actually trending)
    for repo in trending_repos:
        key = repo["full_name"].lower()
        if key not in seen_names:
            seen_names.add(key)
            all_repos.append(repo)
    for repo in search_repos:
        key = repo["full_name"].lower()
        if key not in seen_names:
            seen_names.add(key)
            all_repos.append(repo)

    print(f"\n📊 Total merged GitHub repos: {len(all_repos)}")

    # ── 4. Classify and summarize repos ──
    categorized = {}
    for repo in all_repos:
        cat_id = classify_repo(repo)
        if cat_id not in categorized:
            categorized[cat_id] = []

        summary = generate_summary(repo, "repo")
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)

        categorized[cat_id].append({
            "name": repo.get("full_name") or repo.get("name", ""),
            "url": repo.get("html_url", ""),
            "description": (repo.get("description") or "")[:150],
            "summary_zh": summary,
            "stars": stars,
            "forks": forks,
            "language": repo.get("language") or "",
            "topics": (repo.get("topics") or [])[:6],
            "created_at": repo.get("created_at", ""),
            "source": repo.get("source", "search"),
            "stars_period": repo.get("stars_period", 0),
        })

    MAX_PER_CAT = 4
    for cat_id in categorized:
        categorized[cat_id].sort(key=lambda r: r["stars"], reverse=True)
        categorized[cat_id] = categorized[cat_id][:MAX_PER_CAT]

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

    # ── 5. ArXiv papers ──
    papers = fetch_arxiv_papers()

    paper_topics = {}
    for p in papers:
        topic_id = classify_paper(p)
        if topic_id not in paper_topics:
            paper_topics[topic_id] = []

        summary = generate_summary(p, "paper")

        paper_topics[topic_id].append({
            "title": p["title"],
            "url": p["url"],
            "pdf_url": p.get("pdf_url", ""),
            "abstract": p["abstract"][:200],
            "summary_zh": summary,
            "authors": p.get("authors", [])[:3],
            "categories": p.get("categories", [])[:3],
            "published": p.get("published", ""),
        })

    MAX_PER_TOPIC = 3
    for tid in paper_topics:
        paper_topics[tid] = paper_topics[tid][:MAX_PER_TOPIC]

    topic_lookup = {t["id"]: t for t in PAPER_TOPICS}
    topic_lookup[DEFAULT_PAPER_TOPIC["id"]] = DEFAULT_PAPER_TOPIC

    output_papers = []
    for topic in [*PAPER_TOPICS, DEFAULT_PAPER_TOPIC]:
        if topic["id"] in paper_topics:
            output_papers.append({
                "id": topic["id"],
                "name": topic["name"],
                "papers": paper_topics[topic["id"]],
            })

    # ── 6. Build digest ──
    total_repos = sum(len(c["repos"]) for c in output_categories)
    total_papers = sum(len(t["papers"]) for t in output_papers)
    top_langs = {}
    for c in output_categories:
        for r in c["repos"]:
            if r["language"]:
                top_langs[r["language"]] = top_langs.get(r["language"], 0) + 1
    top3_langs = sorted(top_langs, key=top_langs.get, reverse=True)[:3]

    digest = f"过去 {DAYS_LOOKBACK} 天，共追踪到 {total_repos} 个热门 AI 开源项目"
    if top3_langs:
        digest += f"（主力语言：{', '.join(top3_langs)}）"
    if total_papers:
        digest += f"和 {total_papers} 篇最新 ArXiv 论文"
    digest += "。"
    if output_categories:
        top_cat = max(output_categories, key=lambda c: sum(r["stars"] for r in c["repos"]))
        digest += f"GitHub 最热方向是「{top_cat['name'].split(' ', 1)[1]}」。"

    # ── 7. Output ──
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
        "lookback_days": DAYS_LOOKBACK,
        "digest": digest,
        "categories": output_categories,
        "papers": output_papers,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Written to {args.out}")
    print(f"   {len(output_categories)} repo categories, {total_repos} repos")
    print(f"   {len(output_papers)} paper topics, {total_papers} papers")
    print(f"   Digest: {digest}")


if __name__ == "__main__":
    main()
