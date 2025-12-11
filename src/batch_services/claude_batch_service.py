"""Claude (Anthropic) Batch API service."""
import time
from typing import List, Dict, Any
from anthropic import Anthropic

from src.question_loader import Question
from src.prompts import get_system_message, get_user_message
from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, POLL_INTERVAL


class ClaudeBatchService:
    """Service for handling Claude batch processing."""
    
    def __init__(self, api_key: str = ANTHROPIC_API_KEY, model: str = CLAUDE_MODEL):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.batch_id = None
        self.start_time = None
        self.end_time = None
    
    def create_batch_requests(self, questions: List[Question]) -> List[Dict[str, Any]]:
        """Create batch request objects for Claude."""
        requests = []
        system_msg = get_system_message()
        user_msg = get_user_message()
        
        for question in questions:
            request = {
                "custom_id": question.custom_id,
                "params": {
                    "model": self.model,
                    "max_tokens": 1000,  # Allow full reasoning for research quality
                    "system": system_msg,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": question.get_image_url()
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": user_msg
                                }
                            ]
                        }
                    ]
                }
            }
            requests.append(request)
        
        return requests
    
    def submit_batch(self, questions: List[Question]) -> str:
        """Submit a batch job to Claude."""
        print(f"\n[Claude] Creating batch requests for {len(questions)} questions...")
        requests = self.create_batch_requests(questions)
        
        print("[Claude] Submitting batch job...")
        message_batch = self.client.messages.batches.create(
            requests=requests
        )
        
        self.batch_id = message_batch.id
        self.start_time = time.time()
        print(f"[Claude] Batch submitted: {self.batch_id}")
        print(f"[Claude] Status: {message_batch.processing_status}")
        
        return self.batch_id
    
    def check_status(self) -> str:
        """Check the status of the batch job."""
        if not self.batch_id:
            return "not_submitted"
        
        batch = self.client.messages.batches.retrieve(self.batch_id)
        return batch.processing_status
    
    def wait_for_completion(self, poll_interval: int = POLL_INTERVAL) -> bool:
        """Wait for batch to complete, polling at regular intervals."""
        if not self.batch_id:
            raise ValueError("No batch has been submitted")
        
        print(f"[Claude] Waiting for batch {self.batch_id} to complete...")
        
        while True:
            batch = self.client.messages.batches.retrieve(self.batch_id)
            status = batch.processing_status
            
            elapsed = int(time.time() - self.start_time)
            elapsed_str = f"{elapsed // 60}m {elapsed % 60}s"
            
            print(f"[Claude] Status: {status} (elapsed: {elapsed_str})")
            
            if status == "ended":
                self.end_time = time.time()
                print(f"[Claude] Batch completed successfully!")
                return True
            elif status in ["canceling", "canceled"]:
                print(f"[Claude] Batch {status}!")
                return False
            
            # Still processing
            time.sleep(poll_interval)
    
    def download_results(self) -> List[Dict[str, Any]]:
        """Download and parse batch results."""
        if not self.batch_id:
            raise ValueError("No batch has been submitted")
        
        print(f"[Claude] Retrieving results for batch: {self.batch_id}")
        
        # Retrieve all results
        results = []
        
        # The results() method returns an iterable that automatically handles pagination
        batch_results = self.client.messages.batches.results(self.batch_id)
        
        # Iterate directly over the results
        for result in batch_results:
            # Convert Pydantic result object to dictionary
            result_obj = result.result
            
            # Handle different result types
            if hasattr(result_obj, 'type'):
                if result_obj.type == "succeeded":
                    result_data = {
                        "type": "succeeded",
                        "message": {
                            "content": [{"type": block.type, "text": block.text} for block in result_obj.message.content],
                            "usage": {
                                "input_tokens": result_obj.message.usage.input_tokens,
                                "output_tokens": result_obj.message.usage.output_tokens
                            }
                        }
                    }
                else:
                    # errored, expired, or canceled
                    result_data = {
                        "type": result_obj.type,
                        "error": getattr(result_obj, 'error', None)
                    }
            else:
                result_data = {"type": "unknown"}
            
            result_dict = {
                "custom_id": result.custom_id,
                "result": result_data
            }
            results.append(result_dict)
        
        print(f"[Claude] Retrieved {len(results)} results")
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the batch processing."""
        if not self.batch_id:
            return {}
        
        batch = self.client.messages.batches.retrieve(self.batch_id)
        
        metrics = {
            "batch_id": self.batch_id,
            "status": batch.processing_status,
            "request_counts": {
                "total": batch.request_counts.processing + batch.request_counts.succeeded + batch.request_counts.errored + batch.request_counts.canceled + batch.request_counts.expired,
                "succeeded": batch.request_counts.succeeded,
                "errored": batch.request_counts.errored,
                "canceled": batch.request_counts.canceled,
                "expired": batch.request_counts.expired,
            }
        }
        
        if self.start_time and self.end_time:
            metrics["processing_time_seconds"] = self.end_time - self.start_time
        
        return metrics

