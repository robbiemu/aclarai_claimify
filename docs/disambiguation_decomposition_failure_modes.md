# Disambiguation and Decomposition Failure Modes

This reference collects the negative patterns we want the prospector and dataset generator to surface. Use it when curating training data or tuning prompts, so that the optimizer sees contrastive examples instead of only perfect outputs.

## Disambiguation failure modes

- **Unresolved referent**: Pronoun or demonstrative still points to multiple candidates ("it", "they", "this", "those people"), so the rewritten sentence is not self-contained.
- **Hallucinated detail**: Rewrite injects a name, place, date, or motivation that never appears in the context, breaking factual fidelity.
- **Omitted constraint**: Rewrite drops qualifiers such as hedging, negation, temporal markers, or sarcasm that were present in the original sentence.
- **Formatting drift**: Output is no longer a single sentence (multiple sentences, bullet list, commentary) or escapes the required JSON schema.
- **Unsupported resolution**: Pronoun is swapped for a vague bucket phrase ("the person", "the group") even though a concrete referent exists in context.
- **Confidence mismatch**: Model claims high confidence (>0.8) while the rewrite clearly fails a rule above, making the metadata unreliable.

## Decomposition failure modes

- **Non-atomic claim**: Candidate bundles several distinct assertions into one line or keeps conjunctions that should be split.
- **Missing key claim**: Important segment of the disambiguated sentence never appears in any candidate output.
- **Incorrect node metadata**: Quality flags (`is_atomic`, `is_self_contained`, `is_verifiable`, `passes_criteria`) disagree with the candidate text.
- **Off-topic hallucination**: Candidate introduces concepts that are not present in the disambiguated sentence.
- **Structure violation**: Output JSON omits required fields, returns an empty list without justification, or assigns the wrong `node_type`.
- **Low-information reasoning**: `reasoning` strings are copied boilerplate or blank, providing no guidance for downstream debugging or scoring.

These cases should be explicitly represented in curated negatives so that prompt training rewards corrections instead of memorizing a single ideal answer.

## Curated Prospect Format

Prospector outputs for the disambiguation and decomposition components should follow this structure:

```json
{
  "positive_example": {
    "target_sentence": "It failed with error 500.",
    "context_text": "[0] It failed with error 500.",
    "rationale": "Pronoun requires resolution."
  },
  "negative_examples": [
    {
      "failure_mode": "unresolved_referent",
      "target_sentence": "It failed with error 500.",
      "context_text": "[0] It failed with error 500.",
      "rationale": "Model may leave 'It' unresolved."
    }
  ]
}
```

During dataset generation, pass `--include-negatives` (and optionally `--negative-quota`) to synthesize training records for each failure mode.
