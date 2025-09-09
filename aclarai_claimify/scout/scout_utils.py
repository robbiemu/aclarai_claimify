from typing import Dict, Any, List, Optional
import httpx
import pydantic
import yaml
from datetime import datetime
from ..config import load_claimify_config
from .tools import _safe_request_get, _run_async_safely, RATE_MANAGER, HTTP_CLIENT


def _validate_search_results(results: List[Dict[str, Any]], tool_name: str, tool_args: Dict[str, Any] = None, matching_tool = None) -> Dict[str, Any]:
    # Check if pre-fetching is enabled for this tool
    config = load_claimify_config()
    prefetch_enabled = False
    prefetch_limit = 3
    validate_urls = True
    retry_on_failure = True
    max_retries = 2
    expanded_result_multiplier = 3  # Fetch 3x more results on retry

    if config.scout_agent and config.scout_agent.mission_plan:
        tool_config = config.scout_agent.mission_plan.get_tool_config(tool_name)
        if tool_config:
            prefetch_enabled = tool_config.pre_fetch_pages
            prefetch_limit = tool_config.pre_fetch_limit
            validate_urls = tool_config.validate_urls
            retry_on_failure = tool_config.retry_on_failure
            max_retries = tool_config.max_retries

    # If pre-fetching is not enabled, return results as-is
    if not prefetch_enabled or not validate_urls:
        return {
            "results": results,
            "validation_performed": False,
            "needs_retry": False,
            "filtered_count": 0,
            "retry_performed": False
        }

    # If pre-fetching is enabled, validate URLs and filter out inaccessible ones
    if results and isinstance(results, list):
        # Phase 1: Validate initial batch of results
        print(f"🔍 {tool_name}: Validating initial {min(prefetch_limit, len(results))} results...")
        validated_batch1 = _validate_url_batch(results[:prefetch_limit], tool_name)
        
        # Check if all validated results are bad and we should retry
        accessible_results = [r for r in validated_batch1 if r.get("status") == "accessible"]
        needs_retry = retry_on_failure and len(accessible_results) == 0 and len(results) > 0
        
        if needs_retry and matching_tool and tool_args:
            print(f"🔄 {tool_name}: All initial results inaccessible, auto-retrying with expanded results...")
            
            # Collect URLs that we already know are bad to exclude them
            bad_urls = set()
            for result in validated_batch1:
                if result.get("status") == "inaccessible" and result.get("url"):
                    bad_urls.add(result["url"])
            
            # Try to auto-retry with expanded results
            retry_success = False
            retry_results = []
            
            for attempt in range(max_retries):
                try:
                    # Execute tool with expanded result count
                    expanded_args = tool_args.copy()
                    
                    # Increase result count for retry (if the tool supports it)
                    if "num_results" in expanded_args:
                        expanded_args["num_results"] = expanded_args["num_results"] * expanded_result_multiplier
                    elif "count" in expanded_args:
                        expanded_args["count"] = expanded_args["count"] * expanded_result_multiplier
                    elif "limit" in expanded_args:
                        expanded_args["limit"] = expanded_args["limit"] * expanded_result_multiplier
                    
                    print(f"         🔁 Retry attempt {attempt + 1}/{max_retries} with expanded parameters: {expanded_args}")
                    expanded_tool_result = matching_tool.invoke(expanded_args)
                    
                    if isinstance(expanded_tool_result, dict) and expanded_tool_result.get("status") == "ok":
                        expanded_results = expanded_tool_result.get("results", [])
                        if expanded_results and isinstance(expanded_results, list):
                            # Filter out URLs we already know are bad
                            fresh_results = []
                            for result in expanded_results:
                                url = result.get("url") if isinstance(result, dict) else None
                                if url and url not in bad_urls:
                                    fresh_results.append(result)
                                elif not url:  # Non-URL results are passed through
                                    fresh_results.append(result)
                            
                            if fresh_results:
                                # Validate the fresh results (but limit to avoid excessive requests)
                                fresh_validation_limit = min(prefetch_limit * 2, len(fresh_results))
                                print(f"         🔍 Validating {fresh_validation_limit} fresh results...")
                                validated_fresh = _validate_url_batch(fresh_results[:fresh_validation_limit], tool_name)
                                
                                # Check if we found any accessible results
                                fresh_accessible = [r for r in validated_fresh if r.get("status") == "accessible"]
                                if fresh_accessible:
                                    print(f"         ✅ Retry successful: Found {len(fresh_accessible)} accessible results")
                                    retry_results = validated_fresh
                                    retry_success = True
                                    break
                                else:
                                    print(f"         ⚠️  Retry attempt {attempt + 1} also yielded no accessible results")
                                    # Add newly discovered bad URLs to exclude list
                                    for result in validated_fresh:
                                        if result.get("status") == "inaccessible" and result.get("url"):
                                            bad_urls.add(result["url"])
                            else:
                                print(f"         ⚠️  Retry attempt {attempt + 1} yielded no new results after filtering")
                        else:
                            print(f"         ⚠️  Retry attempt {attempt + 1} returned no results")
                    else:
                        print(f"         ⚠️  Retry attempt {attempt + 1} failed with status: {expanded_tool_result.get('status') if isinstance(expanded_tool_result, dict) else 'unknown'}")
                        
                except Exception as e:
                    print(f"         ❌ Retry attempt {attempt + 1} failed: {e}")
            
            if retry_success:
                # Return the retry results
                return {
                    "results": retry_results,
                    "validation_performed": True,
                    "needs_retry": False,
                    "filtered_count": len([r for r in retry_results if r.get("status") == "inaccessible"]),
                    "original_count": len(results[:prefetch_limit]),
                    "retry_performed": True,
                    "retry_successful": True,
                    "bad_urls_excluded": list(bad_urls)
                }
            else:
                print(f"         ❌ All retry attempts failed, returning validated initial results")
                # Fall back to validated initial results
                return {
                    "results": validated_batch1,
                    "validation_performed": True,
                    "needs_retry": False,  # Don't signal retry since we already tried
                    "filtered_count": len([r for r in validated_batch1 if r.get("status") == "inaccessible"]),
                    "original_count": len(results[:prefetch_limit]),
                    "retry_performed": True,
                    "retry_successful": False,
                    "bad_urls_excluded": list(bad_urls)
                }
        
        # Return initial validation results (either because no retry needed or retry conditions not met)
        filtered_count = len([r for r in validated_batch1 if r.get("status") == "inaccessible"])
        
        return {
            "results": validated_batch1,
            "validation_performed": True,
            "needs_retry": False,
            "filtered_count": filtered_count,
            "original_count": len(results[:prefetch_limit]),
            "retry_performed": False
        }
    
    return {
        "results": results,
        "validation_performed": False,
        "needs_retry": False,
        "filtered_count": 0,
        "retry_performed": False
    }


def _validate_url_batch(results_batch: List[Dict[str, Any]], tool_name: str) -> List[Dict[str, Any]]:
    """
    Validate a batch of search results by checking URL accessibility.
    
    Args:
        results_batch: List of search results to validate
        tool_name: Name of the tool (for logging)
        
    Returns:
        List of validated results with status information
    """
    validated_results = []
    
    for result in results_batch:
        if isinstance(result, dict):
            # Look for URL field - try common field names
            url = None
            for url_field in ["url", "link"]:
                if url_field in result:
                    url = result[url_field]
                    break
            
            if url:
                try:
                    # Test if the URL is accessible with a HEAD request first (more efficient)
                    response = _safe_request_head(url, timeout_s=10, max_retries=1)
                    if response.status_code == 200:
                        result_copy = result.copy()
                        result_copy["status"] = "accessible"
                        validated_results.append(result_copy)
                    else:
                        print(f"⚠️  {tool_name}: URL {url} inaccessible (status {response.status_code})")
                        result_copy = result.copy()
                        result_copy["status"] = "inaccessible"
                        result_copy["error"] = f"HTTP {response.status_code}"
                        validated_results.append(result_copy)
                except Exception as e:
                    # Try GET request as fallback if HEAD fails
                    try:
                        response = _safe_request_get(url, timeout_s=10, max_retries=1)
                        if response.status_code == 200:
                            result_copy = result.copy()
                            result_copy["status"] = "accessible"
                            validated_results.append(result_copy)
                        else:
                            print(f"⚠️  {tool_name}: URL {url} inaccessible (status {response.status_code})")
                            result_copy = result.copy()
                            result_copy["status"] = "inaccessible"
                            result_copy["error"] = f"HTTP {response.status_code}"
                            validated_results.append(result_copy)
                    except Exception as e2:
                        print(f"⚠️  {tool_name}: URL {url} inaccessible ({type(e2).__name__})")
                        result_copy = result.copy()
                        result_copy["status"] = "inaccessible"
                        result_copy["error"] = f"{type(e2).__name__}: {str(e2)}"
                        validated_results.append(result_copy)
            else:
                # No URL field found - check if this might be a non-URL result
                # Look for common fields that indicate this is a valid result without a URL
                has_content_fields = any(field in result for field in ["title", "snippet", "description", "content"])
                if has_content_fields:
                    # This appears to be a valid result without a URL (e.g., Wikipedia summary)
                    result_copy = result.copy()
                    result_copy["status"] = "accessible"
                    validated_results.append(result_copy)
                else:
                    # Non-URL results are passed through as accessible
                    result_copy = result.copy()
                    result_copy["status"] = "accessible"
                    validated_results.append(result_copy)
        else:
            # Non-dict results are passed through as accessible
            validated_results.append({"content": result, "status": "accessible"})
    
    return validated_results


def _safe_request_head(url: str, timeout_s: int = 15, max_retries: int = 2) -> httpx.Response:
    """Efficient HEAD request for URL validation."""
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
                    return await HTTP_CLIENT.head(
                        url, timeout=timeout_s, headers={"User-Agent": "DataScout/1.0"}
                    )

            # Run the coroutine for this specific attempt
            resp = _run_async_safely(do_rate_limited_request())
            return resp
            
        except HTTPStatusError as e:
            # For HEAD requests, some servers return 405 (Method Not Allowed)
            # In that case, we should fall back to GET
            if e.response.status_code == 405:
                raise  # Re-raise to trigger GET fallback in caller
            else:
                last_exc = e
                if attempt < max_retries:
                    import time
                    time.sleep(1.0 * (2 ** attempt))  # Exponential backoff
                continue
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                import time
                time.sleep(1.0 * (2 ** attempt))  # Exponential backoff
            continue
    
    if last_exc:
        raise last_exc
    else:
        raise Exception("HEAD request failed for unknown reasons")


def get_mission_details_from_file(
    mission_plan_path: str,
) -> Optional[Dict[str, Any]]:
    """
    Lightweight parser for mission_config.yaml to get mission names and target sizes
    without instantiating a MissionRunner.

    Args:
        mission_plan_path: Path to the mission_config.yaml file.

    Returns:
        A dictionary containing mission names and their calculated total samples,
        or None if the file cannot be read.
        Structure:
        {
            "mission_names": ["mission1", "mission2"],
            "mission_targets": {
                "mission1": 100,
                "mission2": 200
            }
        }
    """
    try:
        with open(mission_plan_path, "r") as f:
            content = f.read()
            # Remove comment header if it exists
            if content.startswith("#"):
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1 :]
            
            mission_plan = yaml.safe_load(content)
            if not mission_plan or "missions" not in mission_plan:
                return None

            mission_names = []
            mission_targets = {}
            
            for mission in mission_plan.get("missions", []):
                name = mission.get("name")
                if not name:
                    continue
                
                mission_names.append(name)
                
                # Calculate total samples for this mission
                target_size = mission.get("target_size", 0)
                goals = mission.get("goals", [])
                total_samples = target_size * len(goals)
                mission_targets[name] = total_samples
                
            return {
                "mission_names": mission_names,
                "mission_targets": mission_targets
            }

    except (FileNotFoundError, yaml.YAMLError):
        return None