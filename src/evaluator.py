"""Evaluate model responses against ground truth."""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import ANSWERS_FILE, PRICING


class Evaluator:
    """Evaluate model responses and calculate metrics."""
    
    def __init__(self, answers_file: Path = ANSWERS_FILE):
        self.answers_file = answers_file
        self.ground_truth = self.load_ground_truth()
    
    def load_ground_truth(self) -> Dict[str, Dict[str, str]]:
        """Load ground truth answers from JSON file."""
        with open(self.answers_file, "r") as f:
            return json.load(f)
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer letter (A, B, C, D, or E) from response text."""
        if not text:
            return None
        
        # Convert to uppercase for matching
        text = text.upper().strip()
        
        # Look for standalone letter answers
        # Pattern 1: Just the letter (possibly with punctuation)
        match = re.search(r'\b([A-E])\b', text)
        if match:
            return match.group(1)
        
        # Pattern 2: "Answer: A" or "ANSWER A" etc.
        match = re.search(r'ANSWER[:\s]+([A-E])\b', text)
        if match:
            return match.group(1)
        
        # Pattern 3: First occurrence of A, B, C, D, or E
        for char in text:
            if char in 'ABCDE':
                return char
        
        return None
    
    def evaluate_openai_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate OpenAI batch results."""
        evaluations = []
        correct = 0
        total = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        for result in results:
            custom_id = result.get("custom_id")
            
            # Parse subject and question_id from custom_id
            subject, question_id = custom_id.split("_", 1)
            
            # Get ground truth
            ground_truth = self.ground_truth.get(subject, {}).get(question_id)
            
            # Extract model response
            response_obj = result.get("response", {})
            if response_obj.get("status_code") == 200:
                body = response_obj.get("body", {})
                choices = body.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    response_text = message.get("content", "")
                    model_answer = self.extract_answer(response_text)
                    
                    # Track token usage
                    usage = body.get("usage", {})
                    total_input_tokens += usage.get("prompt_tokens", 0)
                    total_output_tokens += usage.get("completion_tokens", 0)
                else:
                    response_text = ""
                    model_answer = None
            else:
                response_text = f"Error: {response_obj.get('status_code')}"
                model_answer = None
            
            # Check correctness
            is_correct = False
            if ground_truth and model_answer:
                is_correct = model_answer.upper() == ground_truth.upper()
                if is_correct:
                    correct += 1
            
            if ground_truth:  # Only count if we have ground truth
                total += 1
            
            evaluations.append({
                "custom_id": custom_id,
                "subject": subject,
                "question_id": question_id,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "response_text": response_text,
                "correct": is_correct
            })
        
        return {
            "evaluations": evaluations,
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "tokens": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_input_tokens + total_output_tokens
            }
        }
    
    def evaluate_claude_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate Claude batch results."""
        evaluations = []
        correct = 0
        total = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        for result in results:
            custom_id = result.get("custom_id")
            
            # Parse subject and question_id from custom_id
            subject, question_id = custom_id.split("_", 1)
            
            # Get ground truth
            ground_truth = self.ground_truth.get(subject, {}).get(question_id)
            
            # Extract model response
            result_obj = result.get("result", {})
            result_type = result_obj.get("type")
            
            if result_type == "succeeded":
                message = result_obj.get("message", {})
                content = message.get("content", [])
                
                # Extract text from content blocks
                response_text = ""
                for block in content:
                    if block.get("type") == "text":
                        response_text += block.get("text", "")
                
                model_answer = self.extract_answer(response_text)
                
                # Track token usage
                usage = message.get("usage", {})
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)
            else:
                response_text = f"Error: {result_type}"
                model_answer = None
            
            # Check correctness
            is_correct = False
            if ground_truth and model_answer:
                is_correct = model_answer.upper() == ground_truth.upper()
                if is_correct:
                    correct += 1
            
            if ground_truth:  # Only count if we have ground truth
                total += 1
            
            evaluations.append({
                "custom_id": custom_id,
                "subject": subject,
                "question_id": question_id,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "response_text": response_text,
                "correct": is_correct
            })
        
        return {
            "evaluations": evaluations,
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "tokens": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_input_tokens + total_output_tokens
            }
        }
    
    def evaluate_gemini_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate Gemini batch results."""
        evaluations = []
        correct = 0
        total = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        for result in results:
            # Gemini batch results use "key" field for custom_id
            custom_id = result.get("key") or result.get("custom_id") or result.get("metadata", {}).get("custom_id")
            
            if not custom_id:
                print(f"[Warning] Skipping result with no custom_id: {result}")
                continue
            
            # Parse subject and question_id from custom_id
            subject, question_id = custom_id.split("_", 1)
            
            # Get ground truth
            ground_truth = self.ground_truth.get(subject, {}).get(question_id)
            
            # Check if this is an error response
            if "error" in result:
                response_text = f"Error: {result.get('error')}"
                model_answer = None
            else:
                # Extract model response
                response_obj = result.get("response", {})
                
                # Gemini response structure
                candidates = response_obj.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    
                    response_text = ""
                    for part in parts:
                        response_text += part.get("text", "")
                    
                    model_answer = self.extract_answer(response_text)
                    
                    # Track token usage if available
                    usage = response_obj.get("usageMetadata", {})
                    total_input_tokens += usage.get("promptTokenCount", 0)
                    total_output_tokens += usage.get("candidatesTokenCount", 0)
                else:
                    response_text = "No response"
                    model_answer = None
            
            # Check correctness
            is_correct = False
            if ground_truth and model_answer:
                is_correct = model_answer.upper() == ground_truth.upper()
                if is_correct:
                    correct += 1
            
            if ground_truth:  # Only count if we have ground truth
                total += 1
            
            evaluations.append({
                "custom_id": custom_id,
                "subject": subject,
                "question_id": question_id,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "response_text": response_text,
                "correct": is_correct
            })
        
        return {
            "evaluations": evaluations,
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "tokens": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_input_tokens + total_output_tokens
            }
        }
    
    def calculate_per_subject_accuracy(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate accuracy breakdown by subject."""
        subjects = {}
        
        for eval_item in evaluations:
            subject = eval_item["subject"]
            
            if subject not in subjects:
                subjects[subject] = {"correct": 0, "total": 0}
            
            if eval_item.get("ground_truth"):  # Only count if we have ground truth
                subjects[subject]["total"] += 1
                if eval_item["correct"]:
                    subjects[subject]["correct"] += 1
        
        # Calculate accuracy percentages
        for subject in subjects:
            total = subjects[subject]["total"]
            subjects[subject]["accuracy"] = subjects[subject]["correct"] / total if total > 0 else 0
        
        return subjects
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Calculate the cost for the given token usage."""
        pricing = PRICING.get(model)
        if not pricing or pricing["input"] is None or pricing["output"] is None:
            return None
        
        # Pricing is per million tokens
        cost = (input_tokens * pricing["input"] / 1_000_000) + \
               (output_tokens * pricing["output"] / 1_000_000)
        
        return cost

