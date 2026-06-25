import argparse
from pathlib import Path
from typing import Any

import yaml


def _flatten_operation_entry(entry: Any) -> Any:
	"""
	Convert an operation entry from:
	  - [{"operation": "insert"}, {"start": 0}, {"end": 10}]
	to:
	  - {"operation": "insert", "start": 0, "end": 10}

	Non-list entries are returned unchanged.
	"""
	if not isinstance(entry, list):
		return entry

	flattened = {}
	for item in entry:
		if isinstance(item, dict):
			flattened.update(item)
	return flattened


def convert_runbook_schema(runbook_obj: dict[str, Any]) -> dict[str, Any]:
	"""
	Convert all datasets in a runbook YAML object to the final_runbook schema.
	"""
	converted: dict[str, Any] = {}

	for dataset_name, dataset_payload in runbook_obj.items():
		if not isinstance(dataset_payload, dict):
			converted[dataset_name] = dataset_payload
			continue

		converted_payload: dict[Any, Any] = {}

		# Keep metadata keys first when present.
		if "max_pts" in dataset_payload:
			converted_payload["max_pts"] = dataset_payload["max_pts"]

		# Convert operation entries.
		for key, value in dataset_payload.items():
			if key in {"max_pts", "gt_url"}:
				continue
			converted_payload[key] = _flatten_operation_entry(value)

		# Keep metadata keys that are typically at the end.
		if "gt_url" in dataset_payload:
			converted_payload["gt_url"] = dataset_payload["gt_url"]

		converted[dataset_name] = converted_payload

	return converted


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Convert runbook YAML to final_runbook-style schema"
	)
	parser.add_argument(
		"-i",
		"--input",
		type=Path,
		required=True,
		help="Path to input runbook YAML (e.g. runbook-100M.yaml)",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		required=True,
		help="Path to output converted YAML",
	)
	args = parser.parse_args()

	print(f"[INFO] Reading input runbook: {args.input}")
	with args.input.open("r", encoding="utf-8") as f:
		runbook_obj = yaml.safe_load(f)

	print("[INFO] Converting schema")
	converted = convert_runbook_schema(runbook_obj)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	print(f"[INFO] Writing converted runbook: {args.output}")
	with args.output.open("w", encoding="utf-8") as f:
		yaml.safe_dump(converted, f, sort_keys=False, default_flow_style=False)

	print("[INFO] Conversion complete")


if __name__ == "__main__":
	main()
