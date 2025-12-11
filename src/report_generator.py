"""Generate benchmark reports in various formats."""
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from src.config import RESULTS_DIR


class ReportGenerator:
    """Generate comprehensive benchmark reports."""
    
    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)
    
    def generate_detailed_json(self, results: Dict[str, Any], filename: str = "detailed_results.json"):
        """Generate detailed JSON report with all evaluations."""
        output_file = self.results_dir / filename
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[Report] Detailed results saved to: {output_file}")
        return output_file
    
    def generate_summary_csv(self, results: Dict[str, Any], filename: str = "summary.csv"):
        """Generate summary CSV with comparison table."""
        summary_data = []
        
        for model_name, model_results in results.items():
            if not isinstance(model_results, dict):
                continue
            
            metrics = model_results.get("metrics", {})
            evaluation = model_results.get("evaluation", {})
            per_subject = model_results.get("per_subject", {})
            
            row = {
                "Model": model_name,
                "Overall Accuracy": f"{evaluation.get('accuracy', 0):.2%}",
                "Correct": evaluation.get('correct', 0),
                "Total": evaluation.get('total', 0),
            }
            
            # Add per-subject accuracies
            for subject, subject_data in per_subject.items():
                row[f"{subject} Accuracy"] = f"{subject_data.get('accuracy', 0):.2%}"
                row[f"{subject} Correct"] = f"{subject_data.get('correct', 0)}/{subject_data.get('total', 0)}"
            
            # Add token usage
            tokens = evaluation.get('tokens', {})
            row["Input Tokens"] = tokens.get('input', 0)
            row["Output Tokens"] = tokens.get('output', 0)
            row["Total Tokens"] = tokens.get('total', 0)
            
            # Add cost if available
            if 'cost' in metrics:
                row["Cost ($)"] = f"${metrics['cost']:.4f}"
            
            # Add processing time
            if 'processing_time_seconds' in metrics:
                time_sec = metrics['processing_time_seconds']
                time_min = int(time_sec // 60)
                time_sec_rem = int(time_sec % 60)
                row["Processing Time"] = f"{time_min}m {time_sec_rem}s"
            
            summary_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        output_file = self.results_dir / filename
        df.to_csv(output_file, index=False)
        
        print(f"[Report] Summary CSV saved to: {output_file}")
        return output_file
    
    def generate_text_report(self, results: Dict[str, Any], filename: str = "benchmark_report.txt"):
        """Generate human-readable text report."""
        output_file = self.results_dir / filename
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("YKS AI MODEL BENCHMARK REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for model_name, model_results in results.items():
                if not isinstance(model_results, dict):
                    continue
                
                f.write(f"\n{model_name.upper()}\n")
                f.write("-" * 80 + "\n\n")
                
                # Overall accuracy
                evaluation = model_results.get("evaluation", {})
                accuracy = evaluation.get('accuracy', 0)
                correct = evaluation.get('correct', 0)
                total = evaluation.get('total', 0)
                
                f.write(f"Overall Performance:\n")
                f.write(f"  Accuracy: {accuracy:.2%} ({correct}/{total} correct)\n\n")
                
                # Per-subject breakdown
                per_subject = model_results.get("per_subject", {})
                if per_subject:
                    f.write(f"Performance by Subject:\n")
                    for subject, subject_data in per_subject.items():
                        subj_acc = subject_data.get('accuracy', 0)
                        subj_correct = subject_data.get('correct', 0)
                        subj_total = subject_data.get('total', 0)
                        f.write(f"  {subject}: {subj_acc:.2%} ({subj_correct}/{subj_total} correct)\n")
                    f.write("\n")
                
                # Token usage
                tokens = evaluation.get('tokens', {})
                f.write(f"Token Usage:\n")
                f.write(f"  Input Tokens: {tokens.get('input', 0):,}\n")
                f.write(f"  Output Tokens: {tokens.get('output', 0):,}\n")
                f.write(f"  Total Tokens: {tokens.get('total', 0):,}\n\n")
                
                # Cost
                metrics = model_results.get("metrics", {})
                if 'cost' in metrics:
                    f.write(f"Cost: ${metrics['cost']:.4f}\n\n")
                
                # Processing time
                if 'processing_time_seconds' in metrics:
                    time_sec = metrics['processing_time_seconds']
                    time_min = int(time_sec // 60)
                    time_sec_rem = int(time_sec % 60)
                    f.write(f"Processing Time: {time_min}m {time_sec_rem}s\n\n")
                
                # Batch info
                if 'batch_id' in metrics:
                    f.write(f"Batch ID: {metrics['batch_id']}\n")
                if 'batch_name' in metrics:
                    f.write(f"Batch Name: {metrics['batch_name']}\n")
                
                f.write("\n")
            
            # Summary comparison
            f.write("=" * 80 + "\n")
            f.write("COMPARISON SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Create comparison table
            model_accuracies = []
            for model_name, model_results in results.items():
                if isinstance(model_results, dict):
                    evaluation = model_results.get("evaluation", {})
                    accuracy = evaluation.get('accuracy', 0)
                    model_accuracies.append((model_name, accuracy))
            
            # Sort by accuracy
            model_accuracies.sort(key=lambda x: x[1], reverse=True)
            
            f.write("Models ranked by accuracy:\n")
            for rank, (model_name, accuracy) in enumerate(model_accuracies, 1):
                f.write(f"  {rank}. {model_name}: {accuracy:.2%}\n")
        
        print(f"[Report] Text report saved to: {output_file}")
        return output_file
    
    def generate_detailed_responses(self, results: Dict[str, Any], filename: str = "detailed_responses.json"):
        """Generate detailed JSON with full model responses/reasoning."""
        output_file = self.results_dir / filename
        
        detailed = {}
        
        for model_name, model_results in results.items():
            if not isinstance(model_results, dict) or model_results.get("status") != "completed":
                continue
            
            evaluation = model_results.get("evaluation", {})
            evaluations = evaluation.get("evaluations", [])
            
            model_responses = []
            for eval_item in evaluations:
                response_data = {
                    "question_id": eval_item.get("custom_id"),
                    "subject": eval_item.get("subject"),
                    "question_number": eval_item.get("question_id"),
                    "ground_truth": eval_item.get("ground_truth"),
                    "model_answer": eval_item.get("model_answer"),
                    "correct": eval_item.get("correct"),
                    "full_response": eval_item.get("response_text"),  # The full reasoning
                }
                model_responses.append(response_data)
            
            detailed[model_name] = {
                "total_questions": len(model_responses),
                "accuracy": evaluation.get("accuracy", 0),
                "responses": model_responses
            }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(detailed, f, indent=2, ensure_ascii=False)
        
        print(f"[Report] Detailed responses saved to: {output_file}")
        return output_file
    
    def generate_all_reports(self, results: Dict[str, Any]):
        """Generate all report formats."""
        print("\n[Report] Generating reports...")
        
        self.generate_detailed_json(results)
        self.generate_detailed_responses(results)  # NEW: Full reasoning
        self.generate_summary_csv(results)
        self.generate_text_report(results)
        
        print("[Report] All reports generated successfully!")

