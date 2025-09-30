"""DSPy signatures for the Claimify pipeline components.

This module defines DSPy Signature classes for each stage of the Claimify pipeline:
Selection, Disambiguation, and Decomposition. These signatures make the inputs and
expected outputs explicit for use with DSPy modules.
"""

from typing import ClassVar

import dspy


class SelectionSignature(dspy.Signature):
    """Signature for the Selection stage of the Claimify pipeline.

    Determines if a sentence contains verifiable factual information suitable for claim extraction.
    """

    instructions: ClassVar[str] = (
        "You are the Selection agent in the Claimify pipeline. Your task is to decide whether a single target sentence contains verifiable factual information suitable for claim extraction.\n\n"
        "Input format\n"
        '- context_text: surrounding text (for context only; use it to disambiguate, not to "rescue" vague sentences).\n'
        "- target_sentence: the specific sentence to evaluate.\n\n"
        "Decision goal\n"
        "Return whether the target_sentence contains at least one concrete, checkable factual claim that can be verified against reliable sources (e.g., official documents, filings, product specs, primary research reports, reproducible measurements, dated press releases). If yes, select it for extraction (downstream components will split compound sentences and verify).\n\n"
        "Core selection criteria\n"
        "Select the sentence if it includes at least one of the following:\n"
        "- Specific, concrete, and checkable facts:\n"
        "  - Numbers with units or clear counts (e.g., 3,140 mAh battery; $7.5 billion; 248 million MAUs; 79 markets; >50 million songs; 2x increase; 35% increase).\n"
        "  - Dates/timeframes (e.g., launched in late 2008; published in 2009; since May 2025).\n"
        "  - Named entities with factual relations (who acquired whom, consideration type; a device can capture photos without an SD card).\n"
        "  - Product/software behavior, defaults, or specifications (e.g., default is Auto; all apps use recommended output; can capture single photos without microSD).\n"
        '  - Document properties that are directly observable (e.g., the PDF is long and structured; has multiple sections/headings; contains numbered findings). Hedged add-ons like "likely tables/footnotes" are acceptable only if at least one other property in the sentence is concrete and checkable.\n'
        "  - Survey/research claims with attribution and concrete figures/sample sizes (e.g., 64% per Gartner; study of 1,749 hotel guests and 1,905 retail consumers).\n"
        "  - Time-bound performance metrics from official/authoritative sources (e.g., since launch, the feature has reduced latency by 15%).\n"
        "- Explicit statements of causality or dependency (e.g., X causes Y; Y is required for Z).\n\n"
        "Rejection criteria\n"
        "Reject the sentence if it is one of the following:\n"
        "- Questions, commands, exclamations, or sentence fragments.\n"
        '- Subjective opinions, marketing claims, or unverifiable qualitative statements (e.g., "easy to use", "beautiful design", "best-in-class").\n'
        '- Vague, non-specific, or purely descriptive statements without a concrete factual anchor (e.g., "something went wrong", "the system is complex", "the report discusses various topics").\n'
        '- Forward-looking statements, predictions, or statements of intent without a concrete, binding commitment (e.g., "we plan to", "the goal is to", "will likely result in"). Exception: select if it is a formal, binding commitment from an authoritative source (e.g., a company filing stating, "we commit to reducing emissions by 50% by 2030").\n'
        "- General statements of principle, definitions, or common knowledge that are not tied to a specific, verifiable event or entity in the text.\n"
        "- Rhetorical or metaphorical language not intended as a literal factual claim.\n\n"
        "Respond with valid JSON only:\n"
        "{\n"
        '  "selected": true/false,\n'
        '  "confidence": 0.0-1.0,\n'
        '  "reasoning": "Brief explanation of decision"\n'
        "}"
    )

    context_text = dspy.InputField(
        desc="Context (surrounding sentences) for the target sentence"
    )
    target_sentence = dspy.InputField(desc="The sentence to evaluate for selection")
    selection_response_json = dspy.OutputField(
        desc="JSON string response with selection decision, confidence, and reasoning"
    )


class DisambiguationSignature(dspy.Signature):
    """Signature for the Disambiguation stage of the Claimify pipeline.

    Rewrites sentences to remove ambiguities and add context.
    """

    instructions: ClassVar[str] = (
        "You are the Disambiguation stage of the Claimify pipeline.\n\n"
        "Goal\n"
        "- Rewrite the target sentence so it is self-contained and unambiguous while preserving the original meaning and scope.\n"
        "- Use only information explicitly available in the provided context_text and target_sentence. Do not introduce external facts, interpretations, or speculative details.\n\n"
        "Inputs\n"
        "- context_text: A short passage that provides antecedents and surrounding details.\n"
        "- target_sentence: The single sentence to be rewritten.\n\n"
        "Output format (JSON only)\n"
        "{\n"
        '  "disambiguated_text": "<a single, self-contained rewritten sentence>",\n'
        '  "changes_made": ["<brief bullet of each minimal change made>"],\n'
        '  "confidence": <float between 0 and 1>\n'
        "}\n\n"
        "Core principles\n"
        "1) Minimal edits, maximum fidelity\n"
        "- Make the smallest set of changes needed to remove ambiguity.\n"
        '- Do not alter the claim’s strength or modality (e.g., do not turn "announcing the discovery" into "announcing a claim").\n'
        "- Preserve tense, tone, and genre (legal, narrative, technical) unless a small, necessary adjustment improves clarity.\n\n"
        "2) Resolve only what the context supports\n"
        '- Replace ambiguous pronouns and demonstratives (it, they, this, that, these, those, here, there, then) with the clearest antecedent explicitly present in context_text (e.g., "the unit," "Atticus Finch," "the forum," "the container," "these political ads" if "political ads" appears in context).\n'
        '- If multiple antecedents are plausible and cannot be resolved from context, prefer a neutral noun phrase already used or directly implied (e.g., "the sender," "the team," "the device") rather than inventing roles, attributes, or identities.\n'
        "- If critical information is omitted (e.g., an ellipsis), do not fabricate or add placeholders; retain the omission and only clarify what can be clarified around it.\n\n"
        "3) Do not add new facts or interpretations\n"
        "- The rewritten sentence must be verifiable using only the provided text.\n"
        "- Do not infer causality, intent, or relationships not explicitly stated.\n\n"
        "Respond with valid JSON only."
    )

    context_text = dspy.InputField(
        desc="Context (surrounding sentences) for the target sentence"
    )
    target_sentence = dspy.InputField(desc="The sentence to disambiguate")
    disambiguation_response_json = dspy.OutputField(
        desc="JSON string response with disambiguated text, changes made, and confidence"
    )


class DecompositionSignature(dspy.Signature):
    """Signature for the Decomposition stage of the Claimify pipeline.

    Breaks disambiguated sentences into atomic, self-contained claims.
    """

    instructions: ClassVar[str] = (
        "You are the Disambiguation stage of the Claimify pipeline.\n\n"
        "Your task is NOT to rewrite by default, but to determine whether the target sentence "
        "contains ambiguity that can be resolved using ONLY the provided context_text.\n\n"
        "Decision rule:\n"
        "- If the target sentence is already clear, self-contained, and unambiguous given the context, "
        "return it EXACTLY as-is.\n"
        "- Only rewrite if there is a specific, resolvable ambiguity (e.g., unclear pronoun, missing antecedent) "
        "that the context explicitly resolves.\n\n"
        "Inputs\n"
        "- context_text: A short passage that may contain antecedents or clarifying details.\n"
        "- target_sentence: The sentence to evaluate and possibly disambiguate.\n\n"
        "Output format (JSON only)\n"
        "{\n"
        '  "disambiguated_text": "<the original sentence if no change needed, otherwise a minimally edited version>",\n'
        '  "changes_made": ["<brief bullet for each change, or empty list if none>"],\n'
        '  "confidence": <float between 0 and 1>\n'
        "}\n\n"
        "Core principles\n"
        "1) Default to NO CHANGE\n"
        "- The safest and most correct output is often the original sentence.\n"
        "- Do not rewrite for style, fluency, emphasis, or perceived 'clarity' unless a genuine ambiguity exists.\n\n"
        "2) Minimal, context-grounded edits only\n"
        "- If you do edit, changes must be minimal and based SOLELY on explicit information in context_text.\n"
        "- Replace ambiguous pronouns/demonstratives (it, they, this, etc.) ONLY with antecedents directly present in context.\n"
        "- Never invent entities, roles, causes, or relationships.\n"
        "- Never add explanatory clauses, definitions, or inferred context.\n\n"
        "3) Preserve meaning and form strictly\n"
        "- Do not alter modality, tense, tone, or claim strength.\n"
        "- Do not retain document-relative referents like ‘the thesis,’ ‘this paper,’ or ‘these methods’ unless they can be grounded in explicit context."
        "- The output must be logically entailed by the original + context—nothing more.\n\n"
        "Remember: If in doubt, DO NOTHING. Return the original sentence with an empty changes_made list.\n"
        "Respond with valid JSON only."
    )

    disambiguated_text = dspy.InputField(
        desc="The disambiguated sentence text to decompose into claims"
    )
    decomposition_response_json = dspy.OutputField(
        desc="JSON string response with claim candidates and their evaluations"
    )
