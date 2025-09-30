# External Dependency Warnings

This document lists warnings that appear during testing but originate from external dependencies that we cannot control or fix directly.

## SWIG-related Deprecation Warnings

**Warning Text:**
```
<frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
<frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
<frozen importlib._bootstrap>:488: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

**Source:** FAISS-CPU package (version 1.12.0)
**Reason:** These warnings originate from the SWIG (Simplified Wrapper and Interface Generator) tool used by FAISS-CPU to create Python bindings. The warnings appear because Python 3.13 has stricter requirements for builtin types to have proper `__module__` attributes, but the SWIG-generated types don't comply with this requirement.

**Why It's Unavoidable:**
1. FAISS-CPU is a required dependency for semantic similarity operations in our dataset generation functionality
2. The warnings come from the C++ extension code that's wrapped with SWIG
3. As users of the FAISS package, we cannot modify its internal SWIG implementation
4. While newer versions of FAISS may address these warnings, upgrading may introduce compatibility issues with our codebase

**Impact:** These warnings are purely informational and do not affect the functionality of our code. The FAISS library works correctly despite these warnings.

## LiteLLM Event Loop Deprecation Warning

**Warning Text:**
```
DeprecationWarning: There is no current event loop
```

**Source:** LiteLLM package (version 1.76.1)
**Reason:** This warning occurs in LiteLLM's async client cleanup code where it attempts to get the current event loop without properly handling cases where no event loop exists.

**Why It's Unavoidable:**
1. LiteLLM is a core dependency for our LLM integration
2. The warning comes from LiteLLM's internal implementation
3. As users of the LiteLLM package, we cannot modify its internal event loop handling

**Impact:** This warning is purely informational and does not affect the functionality of our code. The LiteLLM library works correctly despite this warning. LangSmith tracing will function properly even when this warning appears.

## LiteLLM Custom Logger Initialization Warning (Resolved)

Previously observed when enabling LiteLLM's LangSmith custom logger:
```
LiteLLM:ERROR: litellm_logging.py:3555 - [Non-Blocking Error] Error initializing custom logger: no running event loop
```

We have removed reliance on LiteLLM's custom logger for observability and now use LangChain/LangGraph native tracing via environment variables (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, etc.). As a result, this error should no longer appear during scout runs.

## Resolution Strategy

Rather than suppressing these warnings (which would hide potentially useful information when the underlying libraries are updated), we document them here for transparency. 

If these warnings become problematic in the future, we may consider:
1. Checking for updated versions of FAISS-CPU and LiteLLM that address these warnings
2. Contributing fixes to the upstream projects
3. As a last resort, selectively filtering these specific warnings in our test configuration
