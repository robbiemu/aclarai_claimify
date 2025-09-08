"""
Production-ready tool implementations for the Data Scout Agent.
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

import httpx
import pydantic
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from .litesearch import AsyncRateLimitManager, SearchProviderProxy
from ..config import load_claimify_config


# Optional import for deep crawling
try:
    from libcrawler.libcrawler import crawl_and_convert

    _HAVE_LIBCRAWLER = True
except ImportError:
    crawl_and_convert = None
    _HAVE_LIBCRAWLER = False

# -------------------------
# Globals for tool clients
# -------------------------

# Use a single rate manager and http client for all tools
RATE_MANAGER = AsyncRateLimitManager()
HTTP_CLIENT = httpx.AsyncClient()


def _ensure_event_loop():
    """Ensure we have a running event loop, create one if needed."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        # No event loop running or it's closed, create a new one
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        except Exception:
            # If we can't create a new loop, we'll handle this in the calling code
            raise


def _run_async_safely(coro):
    """Run async code safely, handling event loop issues."""
    try:
        # First try to get current loop
        _loop = asyncio.get_running_loop()
        # We're in an event loop, run in a thread with a new loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=60)
    except RuntimeError:
        # No event loop running, safe to use asyncio.run
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                # Create a new event loop and try again
                _ensure_event_loop()
                return asyncio.run(coro)
            raise


# -------------------------
# Pydantic schemas & tools
# -------------------------


class WebSearchInput(pydantic.BaseModel):
    """Input schema for the web_search tool."""

    query: str = pydantic.Field(description="The search query to execute.")


def create_web_search_tool():
    """
    Factory function that creates a web_search tool with the configured provider hard-coded.
    This ensures that the search provider is determined by the configuration, not by the agent.
    """
    config = load_claimify_config()
    search_provider = "duckduckgo/search"
    if config.scout_agent and config.scout_agent.search_provider:
        search_provider = config.scout_agent.search_provider

    @tool("web_search", args_schema=WebSearchInput)
    def web_search(query: str) -> Dict[str, Any]:
        """
        Performs a web search using the configured provider and returns the results.
        This tool is rate-limited to avoid API abuse.

        Note: This function is synchronous to work with LangGraph's ToolNode.
        It uses asyncio.run() internally to handle the async operations.
        """
        proxy = SearchProviderProxy(
            provider=search_provider,
            rate_limit_manager=RATE_MANAGER,
            http_client=HTTP_CLIENT,
        )
        try:
            # Run the async operation safely with improved event loop management
            results = _run_async_safely(proxy.run(query))

            return {
                "query": query,
                "results": results,
                "provider": search_provider,
                "status": "ok",
            }
        except Exception as e:
            return {
                "query": query,
                "results": None,
                "provider": search_provider,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)}",
            }

    return web_search


class ArxivSearchInput(pydantic.BaseModel):
    """Input schema for the arxiv_search tool."""

    query: str = pydantic.Field(description="The search query to execute on Arxiv.")


@tool("arxiv_search", args_schema=ArxivSearchInput)
def arxiv_search(query: str) -> Dict[str, Any]:
    """
    Performs a search on Arxiv and returns the results.
    This tool is rate-limited to avoid API abuse.

    Note: This function is synchronous to work with LangGraph's ToolNode.
    It uses asyncio.run() internally to handle the async operations.
    """
    proxy = SearchProviderProxy(
        provider="arxiv/search",
        rate_limit_manager=RATE_MANAGER,
        http_client=HTTP_CLIENT,
    )
    try:
        # Run the async operation safely with improved event loop management
        results = _run_async_safely(proxy.run(query))

        return {
            "query": query,
            "results": results,
            "provider": "arxiv/search",
            "status": "ok",
        }
    except Exception as e:
        return {
            "query": query,
            "results": None,
            "provider": "arxiv/search",
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }


class ArxivGetContentInput(pydantic.BaseModel):
    """Input schema for the arxiv_get_content tool."""

    query: str = pydantic.Field(description="The search query to get content for from Arxiv.")


@tool("arxiv_get_content", args_schema=ArxivGetContentInput)
def arxiv_get_content(query: str) -> Dict[str, Any]:
    """
    Retrieves detailed content from Arxiv based on a search query.
    This tool is rate-limited to avoid API abuse.

    Note: This function is synchronous to work with LangGraph's ToolNode.
    It uses asyncio.run() internally to handle the async operations.
    """
    try:
        # Using LangChain's ArxivAPIWrapper for detailed content retrieval
        from langchain_community.utilities import ArxivAPIWrapper
        
        arxiv_wrapper = ArxivAPIWrapper(
            top_k_results=5,
            ARXIV_MAX_QUERY_LENGTH=300,
            load_max_docs=5,
            load_all_available_meta=False,
            doc_content_chars_max=8000
        )
        
        # Get detailed documents
        docs = arxiv_wrapper.load(query)
        
        # Format the results for consistency
        formatted_results = []
        for doc in docs:
            formatted_results.append({
                "title": doc.metadata.get("Title", "N/A"),
                "authors": doc.metadata.get("Authors", "N/A"),
                "published": doc.metadata.get("Published", "N/A"),
                "summary": doc.metadata.get("Summary", "N/A"),
                "content": doc.page_content[:8000] if doc.page_content else "N/A"
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "provider": "arxiv/get_content",
            "status": "ok",
        }
    except Exception as e:
        return {
            "query": query,
            "results": None,
            "provider": "arxiv/get_content",
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }


class WikipediaSearchInput(pydantic.BaseModel):
    """Input schema for the wikipedia_search tool."""

    query: str = pydantic.Field(description="The search query to execute on Wikipedia.")


@tool("wikipedia_search", args_schema=WikipediaSearchInput)
def wikipedia_search(query: str) -> Dict[str, Any]:
    """
    Performs a search on Wikipedia and returns the results.
    This tool is rate-limited to avoid API abuse.

    Note: This function is synchronous to work with LangGraph's ToolNode.
    It uses asyncio.run() internally to handle the async operations.
    """
    proxy = SearchProviderProxy(
        provider="wikipedia/search",
        rate_limit_manager=RATE_MANAGER,
        http_client=HTTP_CLIENT,
    )
    try:
        # Run the async operation safely with improved event loop management
        results = _run_async_safely(proxy.run(query))

        return {
            "query": query,
            "results": results,
            "provider": "wikipedia/search",
            "status": "ok",
        }
    except Exception as e:
        return {
            "query": query,
            "results": None,
            "provider": "wikipedia/search",
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }


class WikipediaGetContentInput(pydantic.BaseModel):
    """Input schema for the wikipedia_get_content tool."""

    query: str = pydantic.Field(description="The search query to get content for from Wikipedia.")


@tool("wikipedia_get_content", args_schema=WikipediaGetContentInput)
def wikipedia_get_content(query: str) -> Dict[str, Any]:
    """
    Retrieves detailed content from Wikipedia based on a search query.
    This tool is rate-limited to avoid API abuse.

    Note: This function is synchronous to work with LangGraph's ToolNode.
    It uses asyncio.run() internally to handle the async operations.
    """
    try:
        # Using LangChain's WikipediaAPIWrapper for detailed content retrieval
        from langchain_community.utilities import WikipediaAPIWrapper
        
        wikipedia_wrapper = WikipediaAPIWrapper(
            top_k_results=3,
            lang="en",
            load_all_available_meta=False,
            doc_content_chars_max=8000
        )
        
        # Get detailed documents
        docs = wikipedia_wrapper.load(query)
        
        # Format the results for consistency
        formatted_results = []
        for doc in docs:
            formatted_results.append({
                "title": doc.metadata.get("title", "N/A"),
                "summary": doc.metadata.get("summary", "N/A"),
                "content": doc.page_content[:8000] if doc.page_content else "N/A"
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "provider": "wikipedia/get_content",
            "status": "ok",
        }
    except Exception as e:
        return {
            "query": query,
            "results": None,
            "provider": "wikipedia/get_content",
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }


# -------------------------
# Utility helpers
# -------------------------

def _truncate_response_for_role(response: Dict[str, Any], role: str) -> Dict[str, Any]:
    """
    Truncate tool responses based on the max_tokens setting for the given role.
    
    Args:
        response: The tool response to potentially truncate
        role: The role of the agent calling the tool
        
    Returns:
        The response, possibly with truncated results
    """
    # Load the configuration to get max_tokens for the role
    config = load_claimify_config()
    
    # Get max_tokens for the role
    max_tokens = None
    if config.scout_agent and config.scout_agent.mission_plan:
        node_config = config.scout_agent.mission_plan.get_node_config(role)
        if node_config:
            max_tokens = node_config.max_tokens
        else:
            # Fallback to default max_tokens from config
            max_tokens = config.max_tokens
    
    # If we couldn't determine max_tokens, use a reasonable default
    if max_tokens is None:
        max_tokens = 1000
    
    # Fields that might contain large text content
    text_fields = ["results", "markdown", "content", "text", "output"]
    
    # Check each text field and truncate if necessary
    for field in text_fields:
        if field in response and isinstance(response[field], str):
            # Rough approximation: assume 1 token ≈ 4 characters
            max_chars = max_tokens * 4
            
            # If the field content is longer than our limit, truncate it
            if len(response[field]) > max_chars:
                truncated = response[field][:max_chars]
                # Add a note that the response was truncated
                response[field] = truncated + f"\n\n[Response truncated to {max_chars} characters due to token limits for role '{role}']"
    
    return response


def _safe_request_get(
    url: str, timeout_s: int = 15, max_retries: int = 2, backoff: float = 1.0
) -> httpx.Response:
    """Retry wrapper around httpx.get with exponential backoff and proper rate limit handling."""

    import urllib.parse
    from httpx import HTTPStatusError

    # Extract domain for rate limiting
    domain = urllib.parse.urlparse(url).netloc

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            # Apply basic rate limiting: wait for domain-specific requests
            # Use the global rate manager to respect per-domain limits
            async def do_rate_limited_request():
                async with RATE_MANAGER.acquire(
                    f"domain:{domain}", requests_per_second=2.0
                ):
                    return await HTTP_CLIENT.get(
                        url, timeout=timeout_s, headers={"User-Agent": "DataScout/1.0"}
                    )

            # Run the coroutine for this specific attempt
            resp = _run_async_safely(do_rate_limited_request())

            # Update rate limit info from headers if available
            RATE_MANAGER.update_from_headers(f"domain:{domain}", resp.headers)

            resp.raise_for_status()
            return resp

        except HTTPStatusError as e:
            last_exc = e

            # Handle 429 (Too Many Requests) with special backoff
            if e.response.status_code == 429:
                # Check for Retry-After header
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                        print(
                            f"      ⏳ Rate limited (429) for {url}, waiting {wait_time}s as instructed"
                        )
                        time.sleep(min(wait_time, 60))  # Cap at 60 seconds
                        continue
                    except ValueError:
                        pass  # Fall back to exponential backoff

                # No retry-after, use longer backoff for rate limits
                if attempt < max_retries:
                    rate_limit_backoff = backoff * (
                        3**attempt
                    )  # More aggressive for 429s
                    print(
                        f"      ⏳ Rate limited (429) for {url}, backing off {rate_limit_backoff:.1f}s"
                    )
                    time.sleep(rate_limit_backoff)
                    continue
                else:
                    print(
                        f"      ❌ Rate limit exceeded for {url} after {max_retries + 1} attempts"
                    )
                    raise

            # Handle other 4xx errors (don't retry client errors except 429)
            elif 400 <= e.response.status_code < 500:
                print(
                    f"      ❌ Client error {e.response.status_code} for {url}, not retrying"
                )
                raise

            # Handle 5xx errors (server errors, worth retrying)
            elif e.response.status_code >= 500:
                if attempt < max_retries:
                    server_error_backoff = backoff * (2**attempt)
                    print(
                        f"      ⚠️ Server error {e.response.status_code} for {url}, retrying in {server_error_backoff:.1f}s"
                    )
                    time.sleep(server_error_backoff)
                    continue
                else:
                    print(
                        f"      ❌ Server error persisted for {url} after {max_retries + 1} attempts"
                    )
                    raise

        except Exception as e:
            last_exc = e
            # For other exceptions (timeouts, connection errors, etc.)
            if attempt < max_retries:
                regular_backoff = backoff * (2**attempt)
                error_type = type(e).__name__
                error_message = str(e)
                print(
                    f"      ⚠️ {error_type} for {url}: {error_message}, retrying in {regular_backoff:.1f}s"
                )
                time.sleep(regular_backoff)
            else:
                error_type = type(e).__name__
                error_message = str(e)
                print(
                    f"      ❌ {error_type} for {url} after {max_retries + 1} attempts: {error_message}"
                )
                raise

    raise last_exc  # pragma: no cover


def _extract_main_text_and_title(
    html: str, css_selector: Optional[str] = None
) -> Dict[str, str]:
    """Extract title and main textual content from HTML."""
    soup = BeautifulSoup(html, "html5lib")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]
    ):
        tag.decompose()
    content_text = soup.get_text(" ", strip=True)
    content_text = re.sub(r"\s+\n", "\n", content_text)
    content_text = re.sub(r"[ \t]{2,}", " ", content_text).strip()
    return {"title": title, "text": content_text}


def _to_markdown_simple(
    title: str, text: str, url: Optional[str] = None, add_front_matter: bool = True
) -> str:
    """Produce a simple Markdown representation."""
    parts = []
    if add_front_matter:
        fm = {
            "source_url": url or "",
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
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

    url: str = pydantic.Field(
        description="Fully qualified URL to fetch (e.g., https://example.com/article)."
    )
    css_selector: Optional[str] = pydantic.Field(
        default=None, description="Optional CSS selector to isolate the main content."
    )
    timeout_s: int = pydantic.Field(default=15, description="HTTP timeout in seconds.")
    max_retries: int = pydantic.Field(default=2, description="Network retry attempts.")
    add_front_matter: bool = pydantic.Field(
        default=True,
        description="If true, include minimal front-matter metadata in the returned Markdown.",
    )


@tool("url_to_markdown", args_schema=UrlToMarkdownInput)
def url_to_markdown(
    url: str,
    css_selector: Optional[str] = None,
    timeout_s: int = 15,
    max_retries: int = 2,
    add_front_matter: bool = True,
) -> Dict[str, Any]:
    """
    Fetch a single web page, extract the main textual content and title, and return a Markdown string plus metadata.
    Returns a dictionary with status, URL, markdown, title, and a text snippet.
    """
    try:
        resp = _safe_request_get(url, timeout_s=timeout_s, max_retries=max_retries)
        html = resp.text
        extracted = _extract_main_text_and_title(html, css_selector=css_selector)
        markdown = _to_markdown_simple(
            extracted["title"],
            extracted["text"],
            url=url,
            add_front_matter=add_front_matter,
        )
        snippet = extracted["text"][:500].strip()
        return {
            "url": url,
            "markdown": markdown,
            "title": extracted["title"],
            "text_snippet": snippet,
            "status": "ok",
        }
    except Exception as e:
        return {
            "url": url,
            "markdown": "",
            "title": "",
            "text_snippet": "",
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }


class CrawlInput(pydantic.BaseModel):
    """Input schema for documentation_crawler tool."""

    base_url: str = pydantic.Field(
        description="The base URL of the documentation site (e.g., https://example.com)."
    )
    starting_point: str = pydantic.Field(
        description="The starting path (e.g., /docs/ or /en/latest/)."
    )
    max_depth: int = pydantic.Field(
        default=3, description="Max crawl depth to avoid runaway crawls."
    )
    allowed_paths: Optional[List[str]] = pydantic.Field(
        default=None,
        description="Optional list of URL paths to include during crawling.",
    )
    ignore_paths: Optional[List[str]] = pydantic.Field(
        default=None, description="Optional list of URL paths to skip during crawling."
    )
    timeout_s: int = pydantic.Field(
        default=30, description="Timeout for network ops during heuristic scanning."
    )
    similarity_threshold: float = pydantic.Field(
        default=0.7, description="Duplicate-similarity threshold for libcrawler."
    )


if _HAVE_LIBCRAWLER:

    @tool("documentation_crawler", args_schema=CrawlInput)
    def documentation_crawler(
        base_url: str,
        starting_point: str,
        max_depth: int = 3,
        allowed_paths: Optional[List[str]] = None,
        ignore_paths: Optional[List[str]] = None,
        timeout_s: int = 30,
        similarity_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Deep-crawl a documentation site using Playwright for JS rendering and return a structured result.
        
        Best practices for configuration:
        1. It's highly recommended to first download and examine the starting page using the url_to_markdown tool
           to understand the site structure before running this crawler.
        2. Use the insights from the starting page to set appropriate allowed_paths and ignore_paths.
        3. Start with a shallow max_depth (1-2) and increase gradually to avoid crawling too much content.
        
        The tool will attempt to infer sensible allowed/ignored paths if none are provided, but manual 
        configuration typically yields better results.
        """
        base_url_clean = base_url.rstrip("/")
        start_url = urljoin(base_url_clean + "/", starting_point.lstrip("/"))

        # Enhanced pre-scan with better path analysis
        try:
            resp = _safe_request_get(start_url, timeout_s=timeout_s, max_retries=1)
            soup = BeautifulSoup(resp.text, "html5lib")
            hrefs = {
                urljoin(start_url, a["href"]) for a in soup.find_all("a", href=True)
            }
            
            # Extract page title for better context
            title_tag = soup.find("title")
            page_title = title_tag.get_text(strip=True) if title_tag else "Unknown"
        except Exception as e:
            return {
                "base_url": base_url,
                "start_url": start_url,
                "pages": {},
                "status": "error",
                "error": f"Pre-scan failed: {type(e).__name__}: {str(e)}",
            }

        # Improved path inference with more sophisticated analysis
        inferred_allowed = allowed_paths or []
        if not inferred_allowed:
            common_prefixes = {}
            path_frequencies = {}
            
            for href in hrefs:
                if href.startswith(base_url_clean):
                    path = urlparse(href).path
                    # Count path segments to identify common structures
                    segments = [seg for seg in path.strip("/").split("/") if seg]
                    if segments:
                        # Track common first segments (e.g., /docs, /api, /guides)
                        first_segment = "/" + segments[0] if segments else "/"
                        common_prefixes[first_segment] = common_prefixes.get(first_segment, 0) + 1
                        
                        # Track path depth frequencies
                        depth = len(segments)
                        path_frequencies[depth] = path_frequencies.get(depth, 0) + 1
            
            # Sort by frequency and take top 3 paths
            if common_prefixes:
                inferred_allowed = [
                    p
                    for p, count in sorted(
                        common_prefixes.items(), key=lambda item: item[1], reverse=True
                    )[:3]
                ]
            else:
                inferred_allowed = [starting_point]

        # Enhanced ignore patterns with more comprehensive defaults
        inferred_ignore = ignore_paths or []
        common_ignores = [
            "/login", "/signup", "/search", "/admin", "/dashboard", 
            "/account", "/profile", "/settings", "/download", "/assets",
            "/static", "/images", "/css", "/js", "/fonts"
        ]
        for ignore in common_ignores:
            if ignore not in inferred_ignore:
                inferred_ignore.append(ignore)

        temp_dir = tempfile.mkdtemp(prefix="data_scout_crawl_")
        output_file = os.path.join(temp_dir, "crawled_docs.md")

        try:
            # Pass additional metadata to libcrawler for better processing
            asyncio.run(
                crawl_and_convert(
                    start_url=start_url,
                    base_url=base_url_clean,
                    output_filename=output_file,
                    allowed_paths=inferred_allowed,
                    ignore_paths=inferred_ignore,
                    similarity_threshold=similarity_threshold,
                    max_depth=max_depth,
                )
            )
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    md_all = f.read()
                summary = md_all[:1000] + ("..." if len(md_all) > 1000 else "")
                pedigree_entry = f"""### {time.strftime("%Y-%m-%d")} — Documentation Crawl: {base_url}
- **Start URL:** `{start_url}`
- **Page Title:** `{page_title}`
- **Max Depth:** `{max_depth}`
- **Similarity Threshold:** `{similarity_threshold}`
- **Allowed Paths (Inferred):** `{json.dumps(inferred_allowed)}`
- **Ignored Paths (Inferred):** `{json.dumps(inferred_ignore)}`"""
                return {
                    "base_url": base_url,
                    "start_url": start_url,
                    "page_title": page_title,
                    "full_markdown": md_all,
                    "summary": summary,
                    "pedigree_entry": pedigree_entry,
                    "status": "ok",
                }
            else:
                return {
                    "base_url": base_url,
                    "start_url": start_url,
                    "pages": {},
                    "status": "error",
                    "error": "libcrawler completed but output file was not created",
                }
        except Exception as e:
            return {
                "base_url": base_url,
                "start_url": start_url,
                "pages": {},
                "status": "error",
                "error": f"crawl_and_convert failed: {type(e).__name__}: {str(e)}",
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
else:
    documentation_crawler = None


class WriteFileInput(pydantic.BaseModel):
    """Input schema for write_file tool."""

    filepath: str = pydantic.Field(
        description="Full path (directory + filename) to write."
    )
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
        return {
            "filepath": filepath,
            "bytes_written": bytes_written,
            "sha256": sha,
            "status": "ok",
        }
    except Exception as e:
        return {
            "filepath": filepath,
            "bytes_written": 0,
            "sha256": None,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }


# -------------------------
# Fallback sample generation
# -------------------------

# A proper synthetic agent would need to be context-aware and generate targeted examples, a tool may not be useful for this.

# -------------------------
# Tool discovery / role scoping
# -------------------------


def get_available_tools() -> List[Callable]:
    """Return actual tool callables available in this environment."""
    # Create the web_search tool with the configured provider
    web_search = create_web_search_tool()
    core_tools = [url_to_markdown, write_file, web_search, arxiv_search, arxiv_get_content, wikipedia_search, wikipedia_get_content]
    optional = [documentation_crawler] if _HAVE_LIBCRAWLER else []
    return core_tools + optional


def get_tools_for_role(role: str) -> List[Callable]:
    """Return tools intended for a specific role."""
    role = (role or "").lower()

    # Create the web_search tool with the configured provider
    web_search = create_web_search_tool()

    all_tools = {
        "web_search": web_search,
        "arxiv_search": arxiv_search,
        "arxiv_get_content": arxiv_get_content,
        "wikipedia_search": wikipedia_search,
        "wikipedia_get_content": wikipedia_get_content,
        "url_to_markdown": url_to_markdown,
        "documentation_crawler": documentation_crawler,
        "write_file": write_file,
    }

    # Define which tools are available for each role
    role_mapping = {
        "research": [
            "web_search",
            "arxiv_search",
            "arxiv_get_content",
            "wikipedia_search",
            "wikipedia_get_content",
            "url_to_markdown",
        ],
        "archive": ["write_file"],
        "supervisor": [],
        "fitness": [],
        "synthetic": [],
    }

    if _HAVE_LIBCRAWLER:
        role_mapping["research"].append("documentation_crawler")

    tool_names_for_role = role_mapping.get(role, [])
    return [all_tools[tool_name] for tool_name in tool_names_for_role if tool_name in all_tools]
