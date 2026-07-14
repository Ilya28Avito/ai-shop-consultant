import json
from pathlib import Path

REPORT = Path.home() / ".local/share/garak/garak_runs/baseline.report.jsonl"
MAX_EXAMPLES_PER_PROBE = 2


def extract_text(value, limit=300):
    """Достаёт текст из строки или из вложенной структуры prompt/output."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        # список строк или список turns
        parts = [extract_text(v, limit) for v in value]
        return " | ".join(p for p in parts if p)[:limit]
    if isinstance(value, dict):
        # частый случай: {"turns": [{"role": ..., "content": {"text": "..."}}]}
        if "text" in value:
            return extract_text(value["text"], limit)
        if "turns" in value:
            return extract_text(value["turns"], limit)
        if "content" in value:
            return extract_text(value["content"], limit)
        # fallback — просто сериализуем
        return json.dumps(value, ensure_ascii=False)[:limit]
    return str(value)[:limit]


hits_by_probe = {}

with open(REPORT, encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("entry_type") != "attempt":
            continue
        if entry.get("status") != 2:
            continue
        detector_results = entry.get("detector_results", {})
        for detector_name, scores in detector_results.items():
            if scores and scores[0] >= 0.5:
                probe = entry.get("probe_classname", "unknown")
                hits_by_probe.setdefault(probe, [])
                if len(hits_by_probe[probe]) < MAX_EXAMPLES_PER_PROBE:
                    outputs = entry.get("outputs") or [""]
                    hits_by_probe[probe].append({
                        "detector": detector_name,
                        "score": scores[0],
                        "prompt": extract_text(entry.get("prompt", "")),
                        "output": extract_text(outputs[0] if outputs else ""),
                    })

for probe, examples in hits_by_probe.items():
    print(f"\n{'='*80}\nPROBE: {probe}\n{'='*80}")
    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i} (detector: {ex['detector']}, score: {ex['score']:.2f}) ---")
        print(f"INPUT:  {ex['prompt']}")
        print(f"OUTPUT: {ex['output']}")