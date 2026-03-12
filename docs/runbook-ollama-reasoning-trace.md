# Runbook: Visualize Ollama Reasoning Model Traces

## Purpose
Query local Ollama reasoning models and extract their thinking traces separately from the final answer.

## Prerequisites
- Ollama running on `localhost:11434`
- At least one reasoning model pulled (see below)

## Reasoning Models Available Locally

| Model | Size | Family | Role in pipeline |
|-------|------|--------|-----------------|
| `magistral:latest` | 24B Q4_K_M | Mistral | MCQ generation + benchmark |
| `hf.co/unsloth/Qwen3-Next-80B-A3B-Thinking-GGUF:UD-Q4_K_XL` | 80B | Qwen3 | MCQ generation (batch) |
| `hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q8_0` | 30B | Nemotron | MCQ generation + benchmark |

## List Available Models

```bash
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

## Query a Reasoning Model (with thinking trace)

Use `"think": true` and `"stream": false` in the chat payload:

```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "magistral:latest",
  "stream": false,
  "think": true,
  "messages": [{"role": "user", "content": "YOUR PROMPT HERE"}]
}' | python3 -c "
import json, sys
data = json.load(sys.stdin)
msg = data.get('message', {})
print('=== THINKING TRACE ===')
print(msg.get('thinking', '(none)'))
print()
print('=== FINAL ANSWER ===')
print(msg.get('content', ''))
"
```

Replace `magistral:latest` with any reasoning model name.

## Response Fields

| Field | Description |
|-------|-------------|
| `message.thinking` | Internal reasoning chain (only present when `think: true`) |
| `message.content` | Final answer shown to the user |
| `message.role` | Always `"assistant"` |

## Model Behaviour Comparison

| | Magistral 24B | Qwen3-Next 80B |
|-|--------------|----------------|
| Trace style | Compact, calculation-focused, self-correcting | Verbose, metacognitive (reflects on user intent) |
| Position bias | Yes — shuffle distractors before evaluation | Yes |
| Thinking tag | `<think>...</think>` stripped by `extract_json()` | Same |

## Notes

- `think: true` is an Ollama-level flag — not all models support it; non-reasoning models return an empty `thinking` field.
- For long generations, use `"stream": true` and parse newline-delimited JSON chunks instead.
- **Position bias:** after MCQ generation, always call `shuffle_distractors()` (`generate_mcq_GPU.py`) before storing or evaluating — both generation models tend to place the correct answer in position `a`.
- The pipeline uses `/api/generate` (not `/api/chat`) for MCQ generation; `<think>` blocks are stripped by `extract_json()` before JSON parsing.

## MCQ Generation via Pipeline (not raw curl)

```python
from api.utils.generate_mcq_GPU import generate_mcq, validate_mcq, shuffle_distractors

raw = generate_mcq(full_prompt, model_name="magistral:latest")
mcq = validate_mcq(raw)
if mcq:
    mcq = shuffle_distractors(mcq)  # neutralise position bias
```

## Reliability Tests

```bash
cd notebooks

# Intra-rater: GPT-4o stability test (submit same MCQs twice)
python reliability_test.py --n 20

# Inter-rater: compute Cohen's κ after manual annotation
python reliability_test.py --kappa
```

## Related Files

| File | Role |
|------|------|
| `notebooks/benchmark_nemotron_vs_magistral.py` | Full generation + evaluation benchmark |
| `notebooks/reliability_test.py` | GPT-4o evaluator reliability validation |
| `docs/benchmark_nemotron_vs_magistral.html` | Benchmark HTML output |
| `src/page/backend/api/utils/generate_mcq_GPU.py` | Generation + shuffle logic |
| `src/page/backend/eval/` | All evaluation metrics |
