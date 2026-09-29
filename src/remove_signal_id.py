"""
Remove signal_id field from finetune_instructions_cleaned.jsonl output.

This script creates a new file (finetune_instructions_no_id.jsonl) with:
1. signal_id removed from each output field
2. Instruction text updated to not mention signal_id/generated_at in schema

The LLM will generate signal_id post-inference via signal_postprocessor.py.
"""

import json
from pathlib import Path


def remove_signal_id(input_path: str, output_path: str) -> None:
    """
    Read a JSONL file and remove 'signal_id' from each output field.
    Also updates instruction to remove schema references to signal_id/generated_at.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    modified_count = 0
    skipped_count = 0

    # Old instruction pattern -> new instruction pattern
    old_schema = 'Schema: {"direction": "CE"|"PE"|"NEUTRAL", "conviction": float 0.0-1.0, "horizon": "intraday"|"next_session", "signal_id": string, "generated_at": string}'
    new_schema = 'Schema: {"direction": "CE"|"PE"|"NEUTRAL", "conviction": float 0.0-1.0, "horizon": "intraday"|"next_session"}'

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)

                # Parse the output JSON string
                output_json = json.loads(record["output"])

                # Remove signal_id if present
                if "signal_id" in output_json:
                    del output_json["signal_id"]
                    modified_count += 1
                else:
                    skipped_count += 1

                # Update instruction to remove signal_id/generated_at from schema
                if old_schema in record["instruction"]:
                    record["instruction"] = record["instruction"].replace(old_schema, new_schema)

                # Write back with modified output
                record["output"] = json.dumps(output_json)
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

            except json.JSONDecodeError as e:
                print(f"Line {line_num}: Invalid JSON - {e}")
                continue
            except KeyError as e:
                print(f"Line {line_num}: Missing key {e}")
                continue

    print(f"Processed {modified_count + skipped_count} records")
    print(f"  - Modified (signal_id removed): {modified_count}")
    print(f"  - Skipped (no signal_id): {skipped_count}")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove signal_id from finetune_instructions JSONL"
    )
    parser.add_argument(
        "--input",
        default="data/finetune_instructions_cleaned.jsonl",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output",
        default="data/finetune_instructions_no_id.jsonl",
        help="Output JSONL file path"
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_path = project_root / args.output if not Path(args.output).is_absolute() else Path(args.output)

    remove_signal_id(str(input_path), str(output_path))
