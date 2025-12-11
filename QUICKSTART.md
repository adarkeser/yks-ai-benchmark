# Quick Start Guide

## 1. Install Dependencies

```bash
pip3 install -r requirements.txt
```

Or with virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure API Keys

Create a `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## 3. Fill in Ground Truth Answers

Edit `answers.json` and fill in the correct answers for each question. Example:

```json
{
  "tyt-tr": {
    "q1": "A",
    "q2": "C"
  },
  "tyt-sos": {
    "q1": "D"
  }
}
```

## 4. Run the Benchmark

Test with a single model first:

```bash
python3 benchmark.py --claude
```

Or run all models:

```bash
python3 benchmark.py --all
```

## 5. Check Results

Results will be saved in the `results/` directory:

- `detailed_results.json` - Full results for each question
- `summary.csv` - Comparison table
- `benchmark_report.txt` - Human-readable report

## Troubleshooting

**Module not found errors**: Install dependencies with `pip3 install -r requirements.txt`

**API key not found errors**: Make sure you've created `.env` and added your API keys

**Questions not loading**: Verify that question images are in `yks_2025/tyt-tr/` and `yks_2025/tyt-sos/`

## Expected Timeline

- Batch submission: 1-2 minutes
- Batch processing: 15-60 minutes (varies by provider)
- Total runtime: 20-60 minutes per model

The script will poll for results every 60 seconds and show progress updates.
