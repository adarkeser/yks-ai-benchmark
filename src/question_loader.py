"""Load and process YKS question images."""
from pathlib import Path
from typing import List, Dict
from urllib.parse import quote

from src.config import QUESTIONS_DIR, GITHUB_IMAGE_BASE_URL


class Question:
    """Represents a single YKS question."""
    
    def __init__(self, subject: str, question_id: str, image_path: Path):
        self.subject = subject
        self.question_id = question_id
        self.image_path = image_path
        self.custom_id = f"{subject}_{question_id}"
    
    def get_image_url(self) -> str:
        """Get GitHub URL for the image."""
        # Format: {base_url}/{subject}/{filename}
        filename = self.image_path.name
        return f"{GITHUB_IMAGE_BASE_URL}/{self.subject}/{quote(filename)}"
    
    def __repr__(self):
        return f"Question({self.subject}, {self.question_id})"


class QuestionLoader:
    """Load all questions from the YKS directory structure."""
    
    def __init__(self, questions_dir: Path = QUESTIONS_DIR):
        self.questions_dir = questions_dir
        self.questions: List[Question] = []
    
    def load_all_questions(self) -> List[Question]:
        """Load all question images from the directory structure."""
        self.questions = []
        
        # Iterate through subject directories
        for subject_dir in sorted(self.questions_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            
            subject_name = subject_dir.name
            
            # Load all image files from the subject directory
            for image_file in sorted(subject_dir.iterdir()):
                if image_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    # Extract question ID from filename (remove extension)
                    question_id = image_file.stem
                    
                    question = Question(
                        subject=subject_name,
                        question_id=question_id,
                        image_path=image_file
                    )
                    self.questions.append(question)
        
        return self.questions
    
    def get_questions_by_subject(self, subject: str) -> List[Question]:
        """Get all questions for a specific subject."""
        return [q for q in self.questions if q.subject == subject]
    
    def get_question_by_id(self, custom_id: str) -> Question:
        """Get a specific question by its custom_id."""
        for q in self.questions:
            if q.custom_id == custom_id:
                return q
        return None
    
    def get_summary(self) -> Dict[str, int]:
        """Returns a dictionary like: {"tyt-tr": 37, "tyt-sos": 20}."""
        summary = {}
        for question in self.questions:
            summary[question.subject] = summary.get(question.subject, 0) + 1
        return summary


if __name__ == "__main__":
    # Test the loader
    loader = QuestionLoader()
    questions = loader.load_all_questions()
    
    print(f"Loaded {len(questions)} questions")
    print("\nSummary by subject:")
    for subject, count in loader.get_summary().items():
        print(f"  {subject}: {count} questions")
    
    if questions:
        print(f"\nFirst question: {questions[0]}")
        print(f"  Custom ID: {questions[0].custom_id}")
        print(f"  Image path: {questions[0].image_path}")

