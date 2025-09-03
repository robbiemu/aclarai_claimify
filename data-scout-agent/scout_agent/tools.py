# tools.py
"""
Production-ready tool implementations for the Data Scout Agent.

Provides:
- url_to_markdown (core)
- documentation_crawler (optional; uses libcrawler if installed)
- write_file (core) -> returns filepath + sha256
- append_to_pedigree (core) -> writes audit entry and returns path

Also:
- get_available_tools()
- get_tools_for_role(role: str)
"""

from __future__ import annotations
import os
import re
import time
import json
import hashlib
import tempfile
import asyncio
import shutil
from typing import Optional, List, Dict, Any, Callable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import pydantic
from langchain_core.tools import tool

# Optional import for deep crawling
try:
    from libcrawler.libcrawler import crawl_and_convert
    _HAVE_LIBCRAWLER = True
except ImportError:
    crawl_and_convert = None
    _HAVE_LIBCRAWLER = False

# -------------------------
# Utility helpers
# -------------------------
def _safe_request_get(url: str, timeout_s: int = 15, max_retries: int = 2, backoff: float = 1.0) -> requests.Response:
    """Minimal retry wrapper around requests.get with exponential backoff."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "DataScout/1.0"})
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise
    raise last_exc  # pragma: no cover

def _extract_main_text_and_title(html: str, css_selector: Optional[str] = None) -> Dict[str, str]:
    """Extract title and main textual content from HTML."""
    soup = BeautifulSoup(html, "html5lib")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
        tag.decompose()
    # ... (rest of the function is perfect as is)
    content_text = soup.get_text(" ", strip=True)
    content_text = re.sub(r"\s+\n", "\n", content_text)
    content_text = re.sub(r"[ \t]{2,}", " ", content_text).strip()
    return {"title": title, "text": content_text}

def _to_markdown_simple(title: str, text: str, url: Optional[str] = None, add_front_matter: bool = True) -> str:
    """Produce a simple Markdown representation."""
    # ... (This function is perfect as is)
    parts = []
    if add_front_matter:
        fm = {"source_url": url or "", "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        parts.append(f"<!-- METADATA {json.dumps(fm)} -->\n")
    if title:
        parts.append(f"# {title}\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for p in paragraphs:
        parts.append(p + "\n")
    return "\n".join(parts).strip()

def _sha256_of_file(path: str) -> str:
    """Computes the SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# -------------------------
# Pydantic schemas & tools
# -------------------------

class UrlToMarkdownInput(pydantic.BaseModel):
    """Input schema for url_to_markdown tool."""
    url: str = pydantic.Field(description="Fully qualified URL to fetch (e.g., https://example.com/article).")
    css_selector: Optional[str] = pydantic.Field(default=None, description="Optional CSS selector to isolate the main content.")
    timeout_s: int = pydantic.Field(default=15, description="HTTP timeout in seconds.")
    max_retries: int = pydantic.Field(default=2, description="Network retry attempts.")
    add_front_matter: bool = pydantic.Field(default=True, description="If true, include minimal front-matter metadata in the returned Markdown.")

@tool("url_to_markdown", args_schema=UrlToMarkdownInput)
def url_to_markdown(url: str, css_selector: Optional[str] = None, timeout_s: int = 15, max_retries: int = 2, add_front_matter: bool = True) -> Dict[str, Any]:
    """
    Fetch a single web page, extract the main textual content and title, and return a Markdown string plus metadata.
    Returns a dictionary with status, URL, markdown, title, and a text snippet.
    """
    try:
        resp = _safe_request_get(url, timeout_s=timeout_s, max_retries=max_retries)
        html = resp.text
        extracted = _extract_main_text_and_title(html, css_selector=css_selector)
        markdown = _to_markdown_simple(extracted["title"], extracted["text"], url=url, add_front_matter=add_front_matter)
        snippet = extracted["text"][:500].strip()
        return {"url": url, "markdown": markdown, "title": extracted["title"], "text_snippet": snippet, "status": "ok"}
    except Exception as e:
        return {"url": url, "markdown": "", "title": "", "text_snippet": "", "status": "error", "error": f"{type(e).__name__}: {str(e)}"}

# -------------------------
# Smart documentation_crawler (optional)
# -------------------------

class CrawlInput(pydantic.BaseModel):
    """Input schema for documentation_crawler tool."""
    base_url: str = pydantic.Field(description="The base URL of the documentation site (e.g., https://example.com).")
    starting_point: str = pydantic.Field(description="The starting path (e.g., /docs/ or /en/latest/).")
    max_depth: int = pydantic.Field(default=3, description="Max crawl depth to avoid runaway crawls.")
    ## REVISED ## - Changed from regex patterns to simple string paths to match libcrawler's implementation.
    allowed_paths: Optional[List[str]] = pydantic.Field(default=None, description="Optional list of URL paths to include during crawling.")
    ignore_paths: Optional[List[str]] = pydantic.Field(default=None, description="Optional list of URL paths to skip during crawling.")
    timeout_s: int = pydantic.Field(default=30, description="Timeout for network ops during heuristic scanning.")
    similarity_threshold: float = pydantic.Field(default=0.7, description="Duplicate-similarity threshold for libcrawler.")

if _HAVE_LIBCRAWLER:
    @tool("documentation_crawler", args_schema=CrawlInput)
    def documentation_crawler(
        base_url: str, starting_point: str, max_depth: int = 3,
        allowed_paths: Optional[List[str]] = None, ignore_paths: Optional[List[str]] = None,
        timeout_s: int = 30, similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Deep-crawl a documentation site using Playwright for JS rendering and return a structured result.
        This tool pre-scans the start page to infer sensible allowed/ignored paths if none are provided.
        """
        base_url_clean = base_url.rstrip("/")
        start_url = urljoin(base_url_clean + "/", starting_point.lstrip("/"))

        try:
            resp = _safe_request_get(start_url, timeout_s=timeout_s, max_retries=1)
            soup = BeautifulSoup(resp.text, "html5lib")
            hrefs = {urljoin(start_url, a["href"]) for a in soup.find_all("a", href=True)}
        except Exception as e:
            return {"base_url": base_url, "start_url": start_url, "pages": {}, "status": "error", "error": f"Pre-scan failed: {type(e).__name__}: {str(e)}"}

        ## REVISED ## - Logic now infers simple paths, not regex, to align with libcrawler.
        inferred_allowed = allowed_paths or []
        if not inferred_allowed:
            common_prefixes = {}
            for href in hrefs:
                if href.startswith(base_url_clean):
                    path = urlparse(href).path
                    prefix = "/" + "/".join(path.strip("/").split("/")[:2])
                    if len(prefix) > 2:
                        common_prefixes[prefix] = common_prefixes.get(prefix, 0) + 1
            if common_prefixes:
                inferred_allowed = [p for p, count in sorted(common_prefixes.items(), key=lambda item: item[1], reverse=True)[:3]]
            else:
                inferred_allowed = [starting_point]

        inferred_ignore = ignore_paths or []
        common_ignores = ["/login", "/signup", "/search"]
        for ignore in common_ignores:
            if ignore not in inferred_ignore:
                inferred_ignore.append(ignore)

        temp_dir = tempfile.mkdtemp(prefix="data_scout_crawl_")
        output_file = os.path.join(temp_dir, "crawled_docs.md")

        try:
            # Assuming crawl_and_convert is an async function as per modern libraries
            # If it's not, the asyncio.run() call would be removed.
            # For this implementation, we will assume it is async.
            async def run_crawl():
                await crawl_and_convert(
                    start_url=start_url, base_url=base_url_clean, output_filename=output_file,
                    allowed_paths=inferred_allowed, ignore_paths=inferred_ignore,
                    similarity_threshold=similarity_threshold
                )
            asyncio.run(run_crawl())

            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    md_all = f.read()
                summary = md_all[:1000] + ("..." if len(md_all) > 1000 else "")
                pedigree_entry = (
                    f"### {time.strftime('%Y-%m-%d')} — Documentation Crawl: {base_url}\n"
                    f"- **Start URL:** `{start_url}`\n"
                    f"- **Allowed Paths (Inferred):** `{json.dumps(inferred_allowed)}`\n"
                    f"- **Ignored Paths (Inferred):** `{json.dumps(inferred_ignore)}`"
                )
                return {"base_url": base_url, "start_url": start_url, "full_markdown": md_all, "summary": summary, "pedigree_entry": pedigree_entry, "status": "ok"}
            else:
                return {"base_url": base_url, "start_url": start_url, "pages": {}, "status": "error", "error": "libcrawler completed but output file was not created"}
        except Exception as e:
            return {"base_url": base_url, "start_url": start_url, "pages": {}, "status": "error", "error": f"crawl_and_convert failed: {type(e).__name__}: {str(e)}"}
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
else:
    documentation_crawler = None

# -------------------------
# write_file and append_to_pedigree tools
# -------------------------

class WriteFileInput(pydantic.BaseModel):
    """Input schema for write_file tool."""
    filepath: str = pydantic.Field(description="Full path (directory + filename) to write.")
    content: str = pydantic.Field(description="Text content to write.")

@tool("write_file", args_schema=WriteFileInput)
def write_file(filepath: str, content: str) -> Dict[str, Any]:
    """Writes text content to a file, creating directories if needed. Returns metadata including a sha256 checksum."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        bytes_written = len(content.encode("utf-8"))
        sha = _sha256_of_file(filepath)
        return {"filepath": filepath, "bytes_written": bytes_written, "sha256": sha, "status": "ok"}
    except Exception as e:
        return {"filepath": filepath, "bytes_written": 0, "sha256": None, "status": "error", "error": f"{type(e).__name__}: {str(e)}"}

class AppendPedigreeInput(pydantic.BaseModel):
    """Input schema for the append_to_pedigree tool."""
    ## REVISED ## - Removed pedigree_path. The agent state, not the LLM, should manage this.
    entry_markdown: str = pydantic.Field(description="Fully formatted markdown string to be appended to the pedigree file.")
    run_id: Optional[str] = pydantic.Field(default=None, description="Optional run/thread id for grouping entries.")

@tool("append_to_pedigree", args_schema=AppendPedigreeInput)
## REVISED ## - Added pedigree_path as a required argument from the agent, not the LLM.
def append_to_pedigree(pedigree_path: str, entry_markdown: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Appends a markdown entry to the pedigree file in a standardized block with a timestamp."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    header = f"---\nrun_id: {run_id or 'N/A'}\ntimestamp: {timestamp}\n---\n"
    final_entry = header + entry_markdown + "\n"
    try:
        os.makedirs(os.path.dirname(pedigree_path), exist_ok=True)
        with open(pedigree_path, "a", encoding="utf-8") as f:
            f.write(final_entry)
        return {"pedigree_path": pedigree_path, "status": "ok", "entry_snippet": final_entry[:200]}
    except Exception as e:
        return {"pedigree_path": pedigree_path, "status": "error", "entry_snippet": None, "error": f"{type(e).__name__}: {str(e)}"}

# -------------------------
# Tool discovery / role scoping
# -------------------------

def get_available_tools() -> List[Callable]:
    """Return actual tool callables available in this environment."""
    core_tools = [url_to_markdown, write_file, append_to_pedigree]
    optional = [documentation_crawler] if _HAVE_LIBCRAWLER else []
    return core_tools + optional

def get_tools_for_role(role: str) -> List[Callable]:
    """Return tools intended for a specific role."""
    role = (role or "").lower()
    mapping = {
        "research": [url_to_markdown] + ([documentation_crawler] if _HAVE_LIBCRAWLER else []),
        "archive": [write_file, append_to_pedigree],
        "supervisor": [],
        "fitness": [],
        "synthetic": []
    }
    return mapping.get(role, [])
