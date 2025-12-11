"""OpenAI Batch API service."""
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI

from src.question_loader import Question
from src.prompts import get_system_message, get_user_message
from src.config import OPENAI_API_KEY, OPENAI_MODEL, POLL_INTERVAL, RESULTS_DIR


class OpenAIBatchService:
    """Service for handling OpenAI batch processing."""
    
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = OPENAI_MODEL):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_id = None
        self.batch_file_id = None
        self.start_time = None
        self.end_time = None
    
    def create_batch_requests(self, questions: List[Question]) -> List[Dict[str, Any]]:
        """Create batch request objects for OpenAI."""
        requests = []
        system_msg = get_system_message()
        user_msg = get_user_message()
        
        for question in questions:
            request = {
                "custom_id": question.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_msg
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_msg
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": question.get_image_url()
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 5000
                }
            }
            requests.append(request)
        
        return requests
    
    def save_batch_file(self, requests: List[Dict[str, Any]]) -> Path:
        """Save batch requests to JSONL file."""
        batch_file = RESULTS_DIR / f"openai_batch_input_{int(time.time())}.jsonl"
        
        with open(batch_file, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")
        
        return batch_file
    
    def submit_batch(self, questions: List[Question]) -> str:
        """Submit a batch job to OpenAI."""
        print(f"\n[OpenAI] Creating batch requests for {len(questions)} questions...")
        requests = self.create_batch_requests(questions)
        
        print("[OpenAI] Saving batch file...")
        batch_file = self.save_batch_file(requests)
        
        print(f"[OpenAI] Uploading batch file: {batch_file}")
        with open(batch_file, "rb") as f:
            file_response = self.client.files.create(
                file=f,
                purpose="batch"
            )
        self.batch_file_id = file_response.id
        print(f"[OpenAI] File uploaded: {self.batch_file_id}")
        
        print("[OpenAI] Submitting batch job...")
        batch_response = self.client.batches.create(
            input_file_id=self.batch_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        self.batch_id = batch_response.id
        self.start_time = time.time()
        print(f"[OpenAI] Batch submitted: {self.batch_id}")
        print(f"[OpenAI] Status: {batch_response.status}")
        
        return self.batch_id
    
    def check_status(self) -> str:
        """Check the status of the batch job."""
        if not self.batch_id:
            return "not_submitted"
        
        batch = self.client.batches.retrieve(self.batch_id)
        return batch.status
    
    def wait_for_completion(self, poll_interval: int = POLL_INTERVAL) -> bool:
        """Wait for batch to complete, polling at regular intervals."""
        if not self.batch_id:
            raise ValueError("No batch has been submitted")
        
        print(f"[OpenAI] Waiting for batch {self.batch_id} to complete...")
        
        while True:
            batch = self.client.batches.retrieve(self.batch_id)
            status = batch.status
            
            elapsed = int(time.time() - self.start_time)
            elapsed_str = f"{elapsed // 60}m {elapsed % 60}s"
            
            print(f"[OpenAI] Status: {status} (elapsed: {elapsed_str})")
            
            if status == "completed":
                self.end_time = time.time()
                print(f"[OpenAI] Batch completed successfully!")
                return True
            elif status in ["failed", "expired", "cancelled"]:
                print(f"[OpenAI] Batch {status}!")
                return False
            
            # Still processing
            time.sleep(poll_interval)
    
    def download_results(self) -> List[Dict[str, Any]]:
        """Download and parse batch results."""
        if not self.batch_id:
            raise ValueError("No batch has been submitted")
        
        batch = self.client.batches.retrieve(self.batch_id)
        
        if batch.status != "completed":
            raise ValueError(f"Batch is not completed (status: {batch.status})")
        
        if not batch.output_file_id:
            raise ValueError("No output file available")
        
        print(f"[OpenAI] Downloading results from file: {batch.output_file_id}")
        
        # Download the output file
        file_response = self.client.files.content(batch.output_file_id)
        output_content = file_response.read()
        
        # Save to disk
        output_file = RESULTS_DIR / f"openai_batch_output_{self.batch_id}.jsonl"
        output_file.write_bytes(output_content)
        print(f"[OpenAI] Results saved to: {output_file}")
        
        # Parse results
        results = []
        for line in output_content.decode().strip().split("\n"):
            if line:
                results.append(json.loads(line))
        
        print(f"[OpenAI] Parsed {len(results)} results")
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the batch processing."""
        if not self.batch_id:
            return {}
        
        batch = self.client.batches.retrieve(self.batch_id)
        
        metrics = {
            "batch_id": self.batch_id,
            "status": batch.status,
            "request_counts": {
                "total": batch.request_counts.total if batch.request_counts else 0,
                "completed": batch.request_counts.completed if batch.request_counts else 0,
                "failed": batch.request_counts.failed if batch.request_counts else 0,
            }
        }
        
        if self.start_time and self.end_time:
            metrics["processing_time_seconds"] = self.end_time - self.start_time
        
        return metrics

