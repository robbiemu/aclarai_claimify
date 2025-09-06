"""
Simple utilities for deduplicating and ranking research results within a single session.

This replaces the complex ResearchCache approach with simpler functions that operate
on the research_session_cache in the state.
"""

import os
import re
import hashlib
import time
from typing import Any, List, Dict, Optional, Set
from urllib.parse import urlparse, urlunparse, parse_qs
from datetime import datetime


def strip_reasoning_block(content: str, tags: list[str] = None) -> str:
    """
    Removes a reasoning block from the beginning of a string if present.

    This function can strip blocks denoted by various tags like <think>,
    <scratchpad>, <reasoning>, etc.

    Args:
        content: The input string.
        tags: A list of tag names to look for. Defaults to a standard list.

    Returns:
        The string with the initial reasoning block removed.
    """
    if tags is None:
        tags = [
            "think",
            "thinking",
            "thought",
            "scratchpad",
            "reasoning",
            "plan",
            "reflection",
            "rationale",
        ]

    # Create a regex 'or' condition by joining the tags with '|'
    # This will match any of the words in the list.
    tag_pattern = "|".join(tags)

    # The main pattern now uses the tag_pattern.
    # - <({tag_pattern})>: Captures the specific tag found (e.g., "scratchpad").
    # - <\/\1>: The backreference \1 ensures the closing tag matches the opening one.
    pattern = rf"^\s*<({tag_pattern})>(.*?)<\/\1>\s*"

    return re.sub(pattern, "", content, count=1, flags=re.DOTALL | re.IGNORECASE)


def get_characteristic_context(task: Dict, mission_config: Dict) -> Optional[str]:
    """
    Finds the definitional context for a characteristic from the mission config.
    """
    if not task or not mission_config:
        return None

    characteristic_name = task.get("characteristic")
    if not characteristic_name:
        return None

    # Search through all missions and goals to find the matching context
    for mission in mission_config.get("missions", []):
        for goal in mission.get("goals", []):
            if goal.get("characteristic") == characteristic_name:
                return goal.get("context")  # Return the context string

    return None  # Return None if no matching characteristic is found


def get_claimify_strategy_block(characteristic: str) -> str:
    """Get the strategic focus block for Claimify characteristics in data prospecting."""
    strategies = {
        "Decontextualization": """
**Strategic Focus for Decontextualization:**
Look for formal, encyclopedic, reference-style text that presents facts in a neutral, standalone manner. Ideal sources include:
- Academic papers with clear factual statements
- Technical documentation with precise specifications
- News articles with objective reporting style
- Reference materials like encyclopedias or handbooks

Avoid sources with:
- Heavy contextual dependencies ("as mentioned above", "this approach")
- Conversational or informal tone
- Opinion pieces or subjective commentary

The best documents will have sentences that can be extracted and understood independently, without needing surrounding context.""",
        "Coverage": """
**Strategic Focus for Coverage:**
Seek data-dense, comprehensive sources that thoroughly cover their subject matter with factual breadth. Ideal sources include:
- Comprehensive reports or surveys
- Statistical summaries and data compilations
- Complete technical specifications
- Thorough news coverage of events
- Academic literature reviews

Avoid sources with:
- Narrow, single-topic focus
- Sparse factual content
- Heavily theoretical or abstract content

The best documents will be rich repositories of diverse, verifiable facts that demonstrate comprehensive coverage of their domain.""",
        "Entailment": """
**Strategic Focus for Entailment:**
Target sources with clear, logical, unambiguous sentence structures that support straightforward factual claims. Ideal sources include:
- Technical manuals with step-by-step processes
- Scientific papers with clear methodology sections
- News reports with direct factual statements
- Educational materials with explicit explanations
- Legal or regulatory documents with precise language

Avoid sources with:
- Complex, multi-clause sentences
- Ambiguous or vague language
- Heavy use of metaphors or figurative language
- Speculative or hypothetical statements

The best documents will have simple, direct sentences where the logical relationship between premise and conclusion is crystal clear.""",
    }
    return strategies.get(
        characteristic,
        f"Look for sources that demonstrate clear {characteristic} characteristics in their writing style and structure.",
    )


def append_to_pedigree(
    pedigree_path: str, entry_markdown: str, run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Appends a markdown entry to the pedigree file in a standardized block with a timestamp.

    Args:
        pedigree_path: The full path to the pedigree file.
        entry_markdown: The markdown content to append to the file.
        run_id: An optional unique identifier for the run.

    Returns:
        A dictionary containing the status of the operation.
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    header = f"---\nrun_id: {run_id or 'N/A'}\ntimestamp: {timestamp}\n---\n"
    final_entry = header + entry_markdown + "\n"
    try:
        # Ensure the directory exists before attempting to write the file
        directory = os.path.dirname(pedigree_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(pedigree_path, "a", encoding="utf-8") as f:
            f.write(final_entry)

        return {
            "pedigree_path": pedigree_path,
            "status": "ok",
            "entry_snippet": final_entry[:200],
        }
    except Exception as e:
        return {
            "pedigree_path": pedigree_path,
            "status": "error",
            "entry_snippet": None,
            "error": f"{type(e).__name__}: {str(e)}",
        }


def normalize_url(url: str) -> str:
    """
    Normalize URL by removing tracking parameters and fragments.

    Args:
        url: Raw URL string

    Returns:
        Normalized URL string
    """
    if not url or url == "unknown":
        return url

    try:
        parsed = urlparse(url)

        # Remove common tracking parameters
        tracking_params = {
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "fbclid",
            "gclid",
            "msclkid",
            "ref",
            "source",
            "campaign",
            "mtm_source",
            "mtm_medium",
            "mtm_campaign",
        }

        # Parse and filter query parameters
        query_params = parse_qs(parsed.query)
        clean_params = {
            k: v for k, v in query_params.items() if k.lower() not in tracking_params
        }

        # Rebuild query string
        clean_query = "&".join(f"{k}={v[0]}" for k, v in clean_params.items())

        # Return normalized URL without fragment
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                clean_query,
                "",  # Remove fragment
            )
        )
    except Exception:
        return url


def normalize_title(title: str) -> str:
    """
    Normalize title for deduplication by removing common variations.

    Args:
        title: Raw title string

    Returns:
        Normalized title string
    """
    if not title:
        return ""

    # Convert to lowercase and strip
    normalized = title.lower().strip()

    # Remove common prefixes/suffixes
    prefixes_to_remove = [
        "the ",
        "a ",
        "an ",
    ]

    suffixes_to_remove = [
        " - wikipedia",
        " | wikipedia",
        " - reddit",
        " | reddit",
        " - medium",
        " | medium",
        " - github",
        " | github",
        " - stack overflow",
        " | stack overflow",
    ]

    for prefix in prefixes_to_remove:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]

    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]

    # Remove extra whitespace and punctuation
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" .,;:")

    return normalized


def get_domain(url: str) -> str:
    """
    Extract domain from URL.

    Args:
        url: URL string

    Returns:
        Domain string
    """
    if not url or url == "unknown":
        return "unknown"

    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return "unknown"


def create_dedupe_hash(result: Dict) -> str:
    """
    Create a deduplication hash for a research result.

    Args:
        result: Research result dictionary

    Returns:
        Hash string for deduplication
    """
    # Use URL if available, otherwise use normalized title + domain
    url = result.get("url", "")
    title = result.get("title", "")
    content = result.get("content", "")

    if url and url != "unknown":
        normalized_url = normalize_url(url)
        return hashlib.md5(normalized_url.encode()).hexdigest()[:16]
    elif title:
        normalized_title = normalize_title(title)
        domain = get_domain(url) if url else "unknown"
        dedupe_key = f"{normalized_title}::{domain}"
        return hashlib.md5(dedupe_key.encode()).hexdigest()[:16]
    else:
        # Fallback to content hash
        content_snippet = content[:200] if content else ""
        return hashlib.md5(content_snippet.encode()).hexdigest()[:16]


def calculate_result_score(
    result: Dict, query: str = "", provider_weights: Dict[str, float] = None
) -> float:
    """
    Calculate a score for a research result.

    Args:
        result: Research result dictionary
        query: Original search query
        provider_weights: Weights for different providers

    Returns:
        Score between 0.0 and 1.0
    """
    score = 0.0

    if provider_weights is None:
        provider_weights = {
            "web_search": 1.0,
            "url_content": 1.5,  # Full page content is more valuable
            "documentation_crawler": 1.3,
        }

    # Base score from provider trust
    source = result.get("source", "web_search")
    base_score = provider_weights.get(source, 1.0)

    # Content length bonus (longer content generally better)
    content = result.get("content", "")
    content_length = len(content)
    if content_length > 5000:
        length_bonus = 0.3
    elif content_length > 2000:
        length_bonus = 0.2
    elif content_length > 500:
        length_bonus = 0.1
    else:
        length_bonus = 0.0

    # Keyword relevance bonus
    relevance_bonus = 0.0
    if query and content:
        query_lower = query.lower()
        content_lower = content.lower()

        # Check for exact query match
        if query_lower in content_lower:
            relevance_bonus += 0.2

        # Check for individual query terms
        query_terms = query_lower.split()
        matching_terms = sum(1 for term in query_terms if term in content_lower)
        if query_terms:
            relevance_bonus += 0.1 * (matching_terms / len(query_terms))

    # Domain trust bonus
    domain = get_domain(result.get("url", ""))
    trusted_domains = {
        "wikipedia.org": 0.2,
        "github.com": 0.15,
        "stackoverflow.com": 0.1,
        "docs.python.org": 0.15,
        "mozilla.org": 0.1,
    }

    domain_bonus = 0.0
    for trusted_domain, bonus in trusted_domains.items():
        if trusted_domain in domain:
            domain_bonus = bonus
            break

    # Recency bonus (newer results slightly preferred)
    recency_bonus = 0.0
    if "timestamp" in result:
        try:
            timestamp = datetime.fromisoformat(result["timestamp"])
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            if age_hours < 1:
                recency_bonus = 0.05
        except Exception:
            pass

    # Calculate final score
    score = base_score + length_bonus + relevance_bonus + domain_bonus + recency_bonus

    # Normalize to 0-1 range
    return min(1.0, max(0.0, score / 3.0))  # Divide by estimated max score


def dedupe_and_rank_results(
    raw_results: List[Dict],
    query: str = "",
    max_results: int = 10,
    max_per_domain: int = 3,
    provider_weights: Dict[str, float] = None,
) -> List[Dict]:
    """
    Deduplicate and rank research results.

    Args:
        raw_results: List of raw research result dictionaries
        query: Original search query for relevance scoring
        max_results: Maximum number of results to return
        max_per_domain: Maximum results per domain
        provider_weights: Weights for different providers

    Returns:
        List of deduped and ranked results
    """
    if not raw_results:
        return []

    # Step 1: Deduplicate
    seen_hashes: Set[str] = set()
    deduped_results = []

    for result in raw_results:
        dedupe_hash = create_dedupe_hash(result)
        if dedupe_hash not in seen_hashes:
            seen_hashes.add(dedupe_hash)
            # Add the hash to the result for debugging
            result_with_hash = result.copy()
            result_with_hash["_dedupe_hash"] = dedupe_hash
            deduped_results.append(result_with_hash)

    # Step 2: Calculate scores
    for result in deduped_results:
        result["_score"] = calculate_result_score(result, query, provider_weights)

    # Step 3: Sort by score (highest first)
    deduped_results.sort(key=lambda x: x["_score"], reverse=True)

    # Step 4: Apply domain limits
    domain_counts = {}
    final_results = []

    for result in deduped_results:
        if len(final_results) >= max_results:
            break

        domain = get_domain(result.get("url", ""))
        domain_count = domain_counts.get(domain, 0)

        if domain_count < max_per_domain:
            domain_counts[domain] = domain_count + 1
            final_results.append(result)

    return final_results


def log_research_session_summary(
    raw_results: List[Dict], final_results: List[Dict], query: str = ""
):
    """
    Log a summary of the research session results.

    Args:
        raw_results: Original raw results
        final_results: Final processed results
        query: Search query
    """
    print("📊 Research Session Summary:")
    print(f"   Query: {query[:100]}..." if len(query) > 100 else f"   Query: {query}")
    print(f"   Raw results collected: {len(raw_results)}")
    print(f"   After deduplication: {len(final_results)}")

    if final_results:
        avg_score = sum(r.get("_score", 0) for r in final_results) / len(final_results)
        print(f"   Average score: {avg_score:.2f}")

        # Show domain distribution
        domains = {}
        for result in final_results:
            domain = get_domain(result.get("url", ""))
            domains[domain] = domains.get(domain, 0) + 1

        print(f"   Domain distribution: {domains}")

        # Show top 3 results
        print("   Top 3 results:")
        for i, result in enumerate(final_results[:3]):
            title = (
                result.get("title", "No title")[:50] + "..."
                if len(result.get("title", "")) > 50
                else result.get("title", "No title")
            )
            score = result.get("_score", 0)
            print(f"     {i + 1}. {title} (score: {score:.2f})")
