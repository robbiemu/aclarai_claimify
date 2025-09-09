# Cached Reuse Improvement - Data Scout Agent

## Overview

This improvement enhances the Data Scout agent to better utilize research work by implementing a cached reuse system. Instead of throwing away all research work after generating one sample, the system can now reuse cached data from the same research session to generate additional samples.

## Key Changes

### New State Fields (state.py)
- `excluded_urls`: List of URLs already used for samples
- `cached_only_mode`: Flag indicating cached-only operation
- `no_search_tools`: Flag disabling search tools
- `allowed_url_whitelist`: URLs permitted in cached-only mode
- `cached_exhausted`: Flag indicating cache exhaustion
- `next_cycle_cached_reuse`: Carryover plan for next cycle

### Mission Runner Updates (mission_runner.py)
- `_initialize_mission_state()`: Initialize cached reuse fields
- `_prepare_cycle_state()`: Handle cached-only cycle setup
- `_update_mission_progress()`: Persist cached reuse state across cycles

### Supervisor Node Enhancements (nodes.py)

#### Post-Archive Evaluation
- After successful archive, supervisor evaluates cached sources
- LLM selects promising unused URLs from research cache with synthetic budget context
- Decision considers both source quality and remaining synthetic budget
- Creates carryover plan for next cycle if sources available
- Always ends cycle (respects recursion limits)

#### Cached-Only Mode Detection
- Deterministic routing when in cached-only mode
- Quota and whitelist validation
- No LLM consultation for next-agent decision

#### Sentinel and Whitelist Enforcement
- Detects "No More Cached Data" sentinel from research
- Enforces URL whitelist in cached-only mode
- Prevents fabricated URLs

### Research Node Modifications (nodes.py)

#### Cached-Only Mode Support
- Detects cached-only constraints
- Emits sentinel when no allowed URLs remain
- Modified system prompt for cached-only operation
- Filters out search tools in cached-only mode

#### Cache Indexing and Selection
- `index_research_cache()`: Normalize cache entries by URL
- `extract_used_urls_from_findings()`: Parse URLs from reports
- `llm_select_cached_sources()`: LLM-based source selection

## New Workflow

### Normal Cycle
1. Research → Fitness → Archive → Supervisor
2. Post-archive: Supervisor evaluates cache and sets carryover
3. Supervisor → END

### Cached Reuse Cycle
1. Supervisor (cached-only mode) → Research (cached-only)
2. Research uses cached data with URL whitelist
3. Research → Supervisor (sentinel or report)
4. If report: Supervisor → Fitness → Archive → Supervisor → END
5. If sentinel: Supervisor → END

## Key Benefits

1. **Better Resource Utilization**: Reuses research work for multiple samples
2. **Respect Recursion Limits**: Each cycle ends properly
3. **Intelligent Source Selection**: LLM evaluates cache quality
4. **Robust Guardrails**: Multiple safety mechanisms prevent loops
5. **Backward Compatible**: Gracefully handles missing cached data

## Configuration

### LLM Selection Schema
```json
{
  "decision": "reuse_cached" | "stop",
  "selected_urls": ["list", "of", "urls"],
  "rationale": "explanation considering both source quality and synthetic budget"
}
```

### LLM Selection Context
The LLM receives comprehensive context including:
- Mission progress (current samples generated)
- Synthetic budget and current synthetic ratio
- Remaining synthetic samples available
- Remaining total work needed
- Quality assessment of cached sources

### Carryover Plan Structure
```json
{
  "active": true,
  "allowed_url_whitelist": ["url1", "url2"],
  "current_task": {"characteristic": "...", "topic": "..."},
  "research_session_cache": [...],
  "rationale": "LLM reasoning"
}
```

## Testing Considerations

- Mock research_session_cache with multiple URLs
- Verify supervisor sets carryover correctly
- Test cached-only routing determinism
- Validate whitelist enforcement
- Confirm sentinel handling
- Check quota satisfaction logic

## Rollout Notes

- Feature is enabled by default
- Monitor supervisor selection rationale logs
- Watch for infinite loops (should be prevented)
- Verify E2E with multi-URL research sessions
- Confirm proper cycle termination
