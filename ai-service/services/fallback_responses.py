"""Pre-written template copy when Groq is unavailable (is_fallback: true in meta)."""

CATEGORISE_REASONING = (
    "The AI classification service is temporarily unavailable. "
    "This item was assigned to the 'other' category for manual review."
)

QUERY_ANSWER = (
    "The risk analysis service is temporarily unable to generate a full answer. "
    "Please retry shortly or contact support. "
    "Source documents may still be listed for your review when retrieval succeeded."
)

REPORT_MARKDOWN = """# Risk report (generated with limited availability)

## Status
The full AI report could not be completed at this time. Use the brief below and source context for a manual follow-up.

## Executive brief
- Review the provided parameters and any attached risk register items.
- Validate against the latest published advisories and internal policy.

## Next steps
1. Re-run report generation when the service is available.
2. Escalate to the risk team if the window is time-sensitive.
"""
