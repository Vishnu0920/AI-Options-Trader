"""
Post-process LLM output to add signal_id and timestamp.

This function takes raw LLM output (without signal_id) and wraps it with:
- signal_id: A unique UUID for each signal
- generated_at: ISO 8601 timestamp of when the signal was generated

The output matches the required schema:
{
    "direction": "CE"|"PE"|"NEUTRAL",
    "conviction": float 0.0-1.0,
    "horizon": "intraday"|"next_session",
    "signal_id": string (UUID),
    "generated_at": string (ISO 8601)
}
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any


def add_signal_metadata(llm_output: dict[str, Any]) -> dict[str, Any]:
    """
    Add signal_id and generated_at timestamp to LLM output.

    Args:
        llm_output: Raw LLM output dict with keys: direction, conviction, horizon

    Returns:
        Complete signal dict with signal_id and generated_at added

    Example:
        >>> raw = {"direction": "CE", "conviction": 0.75, "horizon": "intraday"}
        >>> result = add_signal_metadata(raw)
        >>> assert "signal_id" in result
        >>> assert "generated_at" in result
        >>> assert result["direction"] == "CE"
    """
    # Generate unique signal ID (UUID v4)
    signal_id = str(uuid.uuid4())

    # Get current UTC timestamp in ISO 8601 format
    generated_at = datetime.now(timezone.utc).isoformat()

    # Build complete signal
    signal = {
        "direction": llm_output["direction"],
        "conviction": llm_output["conviction"],
        "horizon": llm_output["horizon"],
        "signal_id": signal_id,
        "generated_at": generated_at
    }

    return signal


def parse_and_enrich(raw_output: str) -> dict[str, Any]:
    """
    Parse raw LLM string output and add metadata.

    Args:
        raw_output: Raw string output from LLM (JSON string)

    Returns:
        Complete signal dict with all fields

    Raises:
        json.JSONDecodeError: If output is not valid JSON
        KeyError: If required fields are missing
    """
    llm_output = json.loads(raw_output)
    return add_signal_metadata(llm_output)


def format_signal(signal: dict[str, Any]) -> str:
    """
    Format a signal dict as a pretty-printed JSON string.

    Args:
        signal: Complete signal dict with all fields

    Returns:
        JSON string with indentation
    """
    return json.dumps(signal, indent=2)


if __name__ == "__main__":
    # Demo usage
    print("Signal Postprocessor Demo")
    print("=" * 50)

    # Example raw LLM output
    raw_llm_output = {
        "direction": "CE",
        "conviction": 0.72,
        "horizon": "intraday"
    }

    print("\nRaw LLM output:")
    print(json.dumps(raw_llm_output, indent=2))

    # Enrich with metadata
    enriched = add_signal_metadata(raw_llm_output)

    print("\nEnriched signal:")
    print(format_signal(enriched))
