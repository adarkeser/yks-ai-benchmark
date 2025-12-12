"""Gemini (Google) Batch API service."""
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from google import genai
from google.genai import types

from src.question_loader import Question
from src.prompts import get_system_message, get_user_message
from src.config import GOOGLE_API_KEY, GEMINI_MODEL, POLL_INTERVAL, RESULTS_DIR


class GeminiBatchService:
    """Service for handling Gemini batch processing."""
    
    def __init__(self, api_key: str = GOOGLE_API_KEY, model: str = GEMINI_MODEL):
        self.client = genai.Client(api_key=api_key)
        self.model_name = f"models/{model}" if not model.startswith("models/") else model
        self.batch_name = None
        self.start_time = None
        self.end_time = None
        self.input_file = None
        self.output_file = None
    
    def create_batch_file(self, questions: List[Question]) -> Path:
        """Create batch input file for Gemini."""
        system_msg = get_system_message()
        user_msg = get_user_message()
        
        # Create JSONL file with requests
        batch_file = RESULTS_DIR / f"gemini_batch_input_{int(time.time())}.jsonl"
        
        with open(batch_file, "w") as f:
            for question in questions:
                # Format according to Gemini batch API docs
                request = {
                    "key": question.custom_id,
                    "request": {
                        "systemInstruction": {
                            "parts": [{"text": system_msg}]
                        },
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "fileData": {
                                            "fileUri": question.get_image_url(),
                                            "mimeType": "image/png"
                                        }
                                    },
                                    {"text": user_msg}
                                ]
                            }
                        ],
                        "generationConfig": {
                            "maxOutputTokens": 2500
                        }
                    }
                }
                f.write(json.dumps(request) + "\n")
        
        return batch_file
    
    def submit_batch(self, questions: List[Question]) -> str:
        """Submit a batch job to Gemini."""
        print(f"\n[Gemini] Creating batch file for {len(questions)} questions...")
        batch_file = self.create_batch_file(questions)
        
        print(f"[Gemini] Uploading batch file: {batch_file}")
        
        # Upload the input file
        uploaded_file = self.client.files.upload(
            file=str(batch_file),
            config=types.UploadFileConfig(
                display_name=f'batch-input-{int(time.time())}',
                mime_type='jsonl'
            )
        )
        self.input_file = uploaded_file
        print(f"[Gemini] File uploaded: {uploaded_file.name}")
        
        print("[Gemini] Creating batch prediction job...")
        
        # Create batch job using the batch API
        batch_job = self.client.batches.create(
            model=self.model_name,
            src=uploaded_file.name,
            config={
                'display_name': f'yks-benchmark-{int(time.time())}'
            }
        )
        
        self.batch_name = batch_job.name
        self.start_time = time.time()
        print(f"[Gemini] Batch submitted: {self.batch_name}")
        print(f"[Gemini] Status: {batch_job.state}")
        
        return self.batch_name
    
    def check_status(self) -> str:
        """Check the status of the batch job."""
        if not self.batch_name:
            return "not_submitted"
        
        batch = self.client.batches.get(name=self.batch_name)
        return batch.state
    
    def wait_for_completion(self, poll_interval: int = POLL_INTERVAL) -> bool:
        """Wait for batch to complete, polling at regular intervals."""
        if not self.batch_name:
            raise ValueError("No batch has been submitted")
        
        print(f"[Gemini] Waiting for batch {self.batch_name} to complete...")
        
        while True:
            batch = self.client.batches.get(name=self.batch_name)
            status = batch.state
            
            elapsed = int(time.time() - self.start_time)
            elapsed_str = f"{elapsed // 60}m {elapsed % 60}s"
            
            print(f"[Gemini] Status: {status} (elapsed: {elapsed_str})")
            
            if status == "JOB_STATE_SUCCEEDED":
                self.end_time = time.time()
                print(f"[Gemini] Batch completed successfully!")
                return True
            elif status in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
                print(f"[Gemini] Batch {status}!")
                return False
            
            # Still processing
            time.sleep(poll_interval)
    
    def download_results(self) -> List[Dict[str, Any]]:
        """Download and parse batch results."""
        if not self.batch_name:
            raise ValueError("No batch has been submitted")
        
        batch = self.client.batches.get(name=self.batch_name)
        
        if batch.state != "JOB_STATE_SUCCEEDED":
            raise ValueError(f"Batch is not completed (status: {batch.state})")
        
        print(f"[Gemini] Downloading results for batch: {self.batch_name}")
        
        # Check if results file is available
        if not hasattr(batch, 'dest') or not batch.dest or not batch.dest.file_name:
            print("[Gemini] No results file found")
            return []
        
        result_file_name = batch.dest.file_name
        print(f"[Gemini] Results file: {result_file_name}")
        
        # Download the results file
        file_content_buffer = self.client.files.download(file=result_file_name)
        
        # Save to disk
        output_path = RESULTS_DIR / f"gemini_batch_output_{int(time.time())}.jsonl"
        output_path.write_bytes(file_content_buffer)
        print(f"[Gemini] Results saved to: {output_path}")
        
        # Parse results
        results = []
        file_content = file_content_buffer.decode('utf-8')
        
        for line in file_content.split('\n'):
            if line.strip():
                parsed_response = json.loads(line)
                results.append(parsed_response)
        
        print(f"[Gemini] Retrieved {len(results)} results")
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the batch processing."""
        if not self.batch_name:
            return {}
        
        batch = self.client.batches.get(name=self.batch_name)
        
        metrics = {
            "batch_name": self.batch_name,
            "status": batch.state,
            "request_counts": {
                "total": batch.batchStats.totalCount if hasattr(batch, "batchStats") else 0,
            }
        }
        
        if self.start_time and self.end_time:
            metrics["processing_time_seconds"] = self.end_time - self.start_time
        
        return metrics

