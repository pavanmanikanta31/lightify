"""CDPS — Confidence-Driven Prompt Shaping.

Concrete prompt templates per Anthropic reviewer's recommendation:
- High confidence (>0.7): direct, concise, suppress hedging
- Medium confidence (0.4-0.7): cite context, note limitations
- Low confidence (<0.4): meta-cognitive reasoning, explicit uncertainty
"""
from __future__ import annotations

from lightify.types import ContextCapsule

HIGH_CONF_TEMPLATE = """\
Answer using the provided context. Be direct and concise.
If the context directly answers the question, state the answer.
Do not speculate beyond what the context supports.

{context}

Question: {query}"""

MED_CONF_TEMPLATE = """\
Answer using the provided context. Where context is strong, be direct.
Where it is thin, note the limitation briefly.
Cite specific context items [N] when making claims.

{context}

Question: {query}"""

LOW_CONF_TEMPLATE = """\
The retrieved context may be incomplete or unreliable.
Step 1: State what the context does and does not cover.
Step 2: Identify any gaps or contradictions in the provided information.
Step 3: Provide your best answer, clearly marking which parts are
  supported by context vs. general knowledge.
Step 4: Rate your confidence (low/medium/high) and explain why.
If you cannot answer reliably, say so explicitly.

{context}
Context confidence: {confidence:.2f}

Question: {query}"""


HIGH_THRESHOLD = 0.70
LOW_THRESHOLD = 0.40


def shape_prompt(capsule: ContextCapsule, query: str) -> str:
    """Select and fill prompt template based on context confidence."""
    conf = capsule.context_confidence

    if conf >= HIGH_THRESHOLD:
        template = HIGH_CONF_TEMPLATE
    elif conf >= LOW_THRESHOLD:
        template = MED_CONF_TEMPLATE
    else:
        template = LOW_CONF_TEMPLATE

    return template.format(
        context=capsule.prompt,
        query=query,
        confidence=conf,
    )


def get_confidence_band(conf: float) -> str:
    """Return the confidence band name."""
    if conf >= HIGH_THRESHOLD:
        return "high"
    elif conf >= LOW_THRESHOLD:
        return "medium"
    return "low"
