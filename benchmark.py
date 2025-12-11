#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main benchmark script for YKS AI model comparison."""
import argparse
import sys
from pathlib import Path

from src.question_loader import QuestionLoader
from src.batch_services.openai_batch_service import OpenAIBatchService
from src.batch_services.claude_batch_service import ClaudeBatchService
from src.batch_services.gemini_batch_service import GeminiBatchService
from src.evaluator import Evaluator
from src.report_generator import ReportGenerator
from src.config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    OPENAI_MODEL,
    CLAUDE_MODEL,
    GEMINI_MODEL
)


def check_api_keys(models: list) -> bool:
    """Check if required API keys are configured."""
    missing_keys = []
    
    if "openai" in models and not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    if "claude" in models and not ANTHROPIC_API_KEY:
        missing_keys.append("ANTHROPIC_API_KEY")
    if "gemini" in models and not GOOGLE_API_KEY:
        missing_keys.append("GOOGLE_API_KEY")
    
    if missing_keys:
        print(f"Error: Missing API keys: {', '.join(missing_keys)}")
        print("Please set them in your .env file or as environment variables.")
        return False
    
    return True


def run_openai_benchmark(questions: list, evaluator: Evaluator) -> dict:
    """Run benchmark for OpenAI."""
    print("\n" + "=" * 80)
    print(f"RUNNING OPENAI BENCHMARK ({OPENAI_MODEL})")
    print("=" * 80)
    
    try:
        service = OpenAIBatchService()
        
        # Submit batch
        service.submit_batch(questions)
        
        # Wait for completion
        success = service.wait_for_completion()
        
        if not success:
            print("[OpenAI] Batch failed!")
            return {"status": "failed"}
        
        # Download results
        results = service.download_results()
        
        # Evaluate
        evaluation = evaluator.evaluate_openai_results(results)
        
        # Get metrics
        metrics = service.get_metrics()
        
        # Calculate cost
        cost = evaluator.calculate_cost(
            OPENAI_MODEL,
            evaluation['tokens']['input'],
            evaluation['tokens']['output']
        )
        if cost is not None:
            metrics['cost'] = cost
        
        # Calculate per-subject accuracy
        per_subject = evaluator.calculate_per_subject_accuracy(evaluation['evaluations'])
        
        return {
            "status": "completed",
            "model": OPENAI_MODEL,
            "evaluation": evaluation,
            "metrics": metrics,
            "per_subject": per_subject
        }
    
    except Exception as e:
        print(f"[OpenAI] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def run_claude_benchmark(questions: list, evaluator: Evaluator) -> dict:
    """Run benchmark for Claude."""
    print("\n" + "=" * 80)
    print(f"RUNNING CLAUDE BENCHMARK ({CLAUDE_MODEL})")
    print("=" * 80)
    
    try:
        service = ClaudeBatchService()
        
        # Submit batch
        service.submit_batch(questions)
        
        # Wait for completion
        success = service.wait_for_completion()
        
        if not success:
            print("[Claude] Batch failed!")
            return {"status": "failed"}
        
        # Download results
        results = service.download_results()
        
        # Evaluate
        evaluation = evaluator.evaluate_claude_results(results)
        
        # Get metrics
        metrics = service.get_metrics()
        
        # Calculate cost
        cost = evaluator.calculate_cost(
            CLAUDE_MODEL,
            evaluation['tokens']['input'],
            evaluation['tokens']['output']
        )
        if cost is not None:
            metrics['cost'] = cost
        
        # Calculate per-subject accuracy
        per_subject = evaluator.calculate_per_subject_accuracy(evaluation['evaluations'])
        
        return {
            "status": "completed",
            "model": CLAUDE_MODEL,
            "evaluation": evaluation,
            "metrics": metrics,
            "per_subject": per_subject
        }
    
    except Exception as e:
        print(f"[Claude] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def run_gemini_benchmark(questions: list, evaluator: Evaluator) -> dict:
    """Run benchmark for Gemini."""
    print("\n" + "=" * 80)
    print(f"RUNNING GEMINI BENCHMARK ({GEMINI_MODEL})")
    print("=" * 80)
    
    try:
        service = GeminiBatchService()
        
        # Submit batch
        service.submit_batch(questions)
        
        # Wait for completion
        success = service.wait_for_completion()
        
        if not success:
            print("[Gemini] Batch failed!")
            return {"status": "failed"}
        
        # Download results
        results = service.download_results()
        
        # Evaluate
        evaluation = evaluator.evaluate_gemini_results(results)
        
        # Get metrics
        metrics = service.get_metrics()
        
        # Calculate cost
        cost = evaluator.calculate_cost(
            GEMINI_MODEL,
            evaluation['tokens']['input'],
            evaluation['tokens']['output']
        )
        if cost is not None:
            metrics['cost'] = cost
        
        # Calculate per-subject accuracy
        per_subject = evaluator.calculate_per_subject_accuracy(evaluation['evaluations'])
        
        return {
            "status": "completed",
            "model": GEMINI_MODEL,
            "evaluation": evaluation,
            "metrics": metrics,
            "per_subject": per_subject
        }
    
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AI models on YKS exam questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --all          Run all models
  python benchmark.py --openai       Run only OpenAI
  python benchmark.py --claude       Run only Claude
  python benchmark.py --gemini       Run only Gemini
  python benchmark.py --openai --claude  Run OpenAI and Claude
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run all models")
    parser.add_argument("--openai", action="store_true", help="Run OpenAI benchmark")
    parser.add_argument("--claude", action="store_true", help="Run Claude benchmark")
    parser.add_argument("--gemini", action="store_true", help="Run Gemini benchmark")
    parser.add_argument("--limit", type=int, help="Limit to first N questions (for testing)")
    
    args = parser.parse_args()
    
    # Determine which models to run
    models = []
    if args.all:
        models = ["openai", "claude", "gemini"]
    else:
        if args.openai:
            models.append("openai")
        if args.claude:
            models.append("claude")
        if args.gemini:
            models.append("gemini")
    
    if not models:
        parser.print_help()
        sys.exit(1)
    
    # Check API keys
    if not check_api_keys(models):
        sys.exit(1)
    
    print("=" * 80)
    print("YKS AI MODEL BENCHMARK")
    print("=" * 80)
    print(f"Models to benchmark: {', '.join(models)}")
    
    # Load questions
    print("\n[Setup] Loading questions...")
    loader = QuestionLoader()
    questions = loader.load_all_questions()
    
    # Apply limit if specified
    if args.limit:
        questions = questions[:args.limit]
        print(f"[Setup] LIMITED to first {len(questions)} questions (--limit {args.limit})")
    else:
        print(f"[Setup] Loaded {len(questions)} questions")
    
    summary = loader.get_summary()
    for subject, count in summary.items():
        print(f"  - {subject}: {count} questions")
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Run benchmarks
    results = {}
    
    if "openai" in models:
        results["OpenAI"] = run_openai_benchmark(questions, evaluator)
    
    if "claude" in models:
        results["Claude"] = run_claude_benchmark(questions, evaluator)
    
    if "gemini" in models:
        results["Gemini"] = run_gemini_benchmark(questions, evaluator)
    
    # Generate reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    
    report_gen = ReportGenerator()
    report_gen.generate_all_reports(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    
    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and model_results.get("status") == "completed":
            evaluation = model_results.get("evaluation", {})
            accuracy = evaluation.get("accuracy", 0)
            correct = evaluation.get("correct", 0)
            total = evaluation.get("total", 0)
            print(f"{model_name}: {accuracy:.2%} ({correct}/{total})")
    
    print("\nCheck the 'results/' directory for detailed reports.")


if __name__ == "__main__":
    main()

