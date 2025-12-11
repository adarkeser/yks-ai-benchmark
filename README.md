# YKS AI Model Benchmark

Batch benchmark ChatGPT-5, Claude Opus 4.5, and Gemini 3 Pro on Turkish university entrance exam (YKS) questions using each provider's batch API.

## Features

- **Batch Processing**: Uses official batch APIs (50% cost savings)
- **Multiple Models**: OpenAI, Claude, Gemini
- **Comprehensive Metrics**: Accuracy, cost, latency, token usage
- **Per-Subject Analysis**: Breakdown by subject (tyt-tr, tyt-sos)
- **Multiple Report Formats**: JSON, CSV, and text reports

## Setup

1. Install dependencies:
```bash
pip3 install -r requirements.txt
```

2. Configure API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

3. Add ground truth answers - Edit `answers.json` and fill in the correct answers for each question.

## Usage

Run all models:
```bash
python3 benchmark.py --all
```

Run specific models:
```bash
python3 benchmark.py --openai
python3 benchmark.py --claude
python3 benchmark.py --gemini
```

Run multiple models:
```bash
python3 benchmark.py --openai --claude
```

## Output

After running, check the `results/` directory for:

- `detailed_results.json` - Per-question responses and correctness
- `summary.csv` - Comparison table with all metrics
- `benchmark_report.txt` - Human-readable summary

## Pricing (Batch API - 50% Discount)

| Model | Input | Output |
|-------|-------|--------|
| ChatGPT-5 | TBD | TBD |
| Claude Opus 4.5 | $2.50/MTok | $12.50/MTok |
| Gemini 3 Pro | TBD | TBD |

## Expected Runtime

Batch processing typically completes in 15-60 minutes depending on queue and volume.

## Notes

- Ground truth answers must be filled in `answers.json` before running
- The script will poll for batch completion every 60 seconds
- Results are saved to `results/` directory
- Original batch files are preserved for debugging
