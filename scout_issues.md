# Scout Agent Issues - September 6, 2025

## Issue #1: Incorrect Synthetic Provenance Classification

**Log Entry:**
```
[21:35:21] ℹ️ Supervisor: No source URL or research cache found, defaulting provenance to 'synthetic'.
```

**Problem:** Legitimate BBC article research marked as synthetic
**Impact:** Audit trail integrity, training data quality, mission metrics
**Root Cause:** Supervisor fails to resolve source URL from research report

**Details:** Even though the research report correctly identified a BBC source URL, the supervisor couldn't find this URL in the research cache or session cache when determining provenance, so it defaulted to "synthetic". This causes legitimate research data to be misclassified.

---

## Issue #2: Checkpoint Serialization Error

**Log Entry:**
```
[21:35:33] 🏁 Graph Router: Routing to END
[21:35:33] ⚠️  Warning: Could not save state to checkpoint: Object of type HumanMessage is not JSON serializable
```

**Problem:** LangGraph state contains HumanMessage objects that can't be serialized to JSON
**Impact:** Checkpointing fails, state persistence unreliable
**Root Cause:** HumanMessage objects contain non-serializable attributes

**Details:** This serialization error prevents proper state checkpointing, which is the foundation of the resume/retry functionality.

---

## Issue #3: Unexpected Immediate Research Response

**Log Pattern:**
```
[21:35:33] -> Executing Node: SUPERVISOR
- Output: {'next_agent': 'research', ...}
[21:35:33] -> Executing Node: RESEARCH
- Responded: # Data Prospecting Report  <- IMMEDIATE RESPONSE!
```

**Problem:** Research node immediately responds with complete report instead of starting fresh research
**Impact:** Stale data reuse, incorrect sample generation
**Root Cause:** Incomplete state reset due to checkpoint corruption

**Details:** After successful archive, the mission runner should start a fresh research cycle. Instead, the research node appears to replay a previous successful session, suggesting corrupted state restoration.

---

## Issue #4: Incorrect Progress Tracking and Mission Abort

**Log Entries:**
```
[21:35:33] 📊 Progress: 0/6 samples (0.0%)>>    -> Loaded state from checkpoint
[21:35:36] ❗ No user question found; returning request for clarification.
```

**Problem:** After archiving 1 sample, progress shows 0/6 and mission aborts
**Impact:** Mission termination, lost progress, unreliable execution
**Root Cause:** Cascade failure from checkpoint serialization error

**Details:** The serialization error causes:
1. Incomplete state restoration 
2. Progress counter not updated after archive
3. Missing user question for next research cycle
4. Mission abort instead of continuing to next sample

**Root Cause Chain:**
Checkpoint Serialization Failure → Corrupted State Restoration → 
Stale Session Replay → Incorrect Progress Count → 
Missing User Question → Mission Abort