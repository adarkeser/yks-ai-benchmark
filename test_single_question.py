import os
import argparse
from dotenv import load_dotenv

from src.question_loader import QuestionLoader
from src.prompts import get_system_message, get_user_message

# Load environment
load_dotenv()

def test_openai(question):
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    print("\n" + "="*80)
    print("TESTING OPENAI")
    print("="*80)
    
    try:
        client = OpenAI(api_key=api_key)
        
        system_msg = get_system_message()
        user_msg = get_user_message()
        image_url = question.get_image_url()
        
        print(f"\nQuestion: {question.custom_id}")
        print(f"Image URL: {image_url}")
        print(f"\nSystem: {system_msg[:100]}...")
        print(f"User: {user_msg}")
        print("\nSending request...")
        
        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[
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
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=5000
        )
        
        answer = response.choices[0].message.content
        print(f"\n✅ SUCCESS!")
        print(f"\nExtracted Answer: {answer}")
        print(f"Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}")
        print(f"\n{'='*80}")
        print("FULL RESPONSE:")
        print(f"{'='*80}")
        print(response)
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")


def test_claude(question):
    from anthropic import Anthropic
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    print("\n" + "="*80)
    print("TESTING CLAUDE")
    print("="*80)
    
    try:
        client = Anthropic(api_key=api_key)
        
        system_msg = get_system_message()
        user_msg = get_user_message()
        image_url = question.get_image_url()
        
        print(f"\nQuestion: {question.custom_id}")
        print(f"Image URL: {image_url}")
        print(f"\nSystem: {system_msg[:100]}...")
        print(f"User: {user_msg}")
        print("\nSending request...")
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=5000,
            system=system_msg,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": user_msg
                        }
                    ]
                }
            ]
        )
        
        answer = response.content[0].text
        print(f"\n✅ SUCCESS!")
        print(f"\nExtracted Answer: {answer}")
        print(f"Tokens - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")
        print(f"\n{'='*80}")
        print("FULL RESPONSE:")
        print(f"{'='*80}")
        print(response)
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")


def test_gemini(question):
    from google import genai
    
    api_key = os.getenv("GOOGLE_API_KEY")
    print("\n" + "="*80)
    print("TESTING GEMINI")
    print("="*80)
    
    try:
        client = genai.Client(api_key=api_key)
        
        system_msg = get_system_message()
        user_msg = get_user_message()
        image_url = question.get_image_url()
        
        print(f"\nQuestion: {question.custom_id}")
        print(f"Image URL: {image_url}")
        print(f"\nSystem: {system_msg[:100]}...")
        print(f"User: {user_msg}")
        print("\nSending request...")
        
        # Gemini API structure
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "parts": [
                        {
                            "file_data": {
                                "file_uri": image_url,
                                "mime_type": "image/png"
                            }
                        },
                        {
                            "text": user_msg
                        }
                    ]
                }
            ],
            config={
                "system_instruction": system_msg,
                "max_output_tokens": 5000
            }
        )
        
        answer = response.text
        print(f"\n✅ SUCCESS!")
        print(f"\nExtracted Answer: {answer}")
        if hasattr(response, 'usage_metadata'):
            print(f"Tokens - Input: {response.usage_metadata.prompt_token_count}, Output: {response.usage_metadata.candidates_token_count}")
        print(f"\n{'='*80}")
        print("FULL RESPONSE:")
        print(f"{'='*80}")
        print(response)
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test a single question with AI APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_single_question.py --list                    # List all questions
  python test_single_question.py --index 0                 # Test first question
  python test_single_question.py --id tyt-tr_q1            # Test by question ID
  python test_single_question.py --id tyt-tr_q1 --openai  # Test only OpenAI
        """
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available questions and exit"
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Question index to test (0-based)"
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Question ID to test (e.g., tyt-tr_q1)"
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Test OpenAI only"
    )
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Test Claude only"
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Test Gemini only"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("SINGLE QUESTION TEST")
    print("="*80)
    
    loader = QuestionLoader()
    questions = loader.load_all_questions()
    
    if not questions:
        print("❌ No questions found!")
        return
    
    if args.list:
        print(f"\n📋 Available Questions ({len(questions)} total):\n")
        for i, q in enumerate(questions):
            print(f"  [{i:2d}] {q.custom_id:20s} ({q.subject})")
        print("\nUse --index N or --id QUESTION_ID to test a specific question")
        return
    
    question = None
    if args.id:
        for q in questions:
            if q.custom_id == args.id:
                question = q
                break
        if not question:
            print(f"❌ Question '{args.id}' not found!")
            print("Use --list to see available questions")
            return
    elif args.index is not None:
        if 0 <= args.index < len(questions):
            question = questions[args.index]
        else:
            print(f"❌ Index {args.index} out of range (0-{len(questions)-1})")
            return
    else:
        question = questions[0]
        print("ℹ️  No question specified, using first question (use --list to see all)")
    
    print(f"\n📊 Testing Question:")
    print(f"   ID: {question.custom_id}")
    print(f"   Subject: {question.subject}")
    print(f"   Image: {question.image_path.name}")
    
    test_all = not (args.openai or args.claude or args.gemini)
    
    if test_all or args.openai:
        test_openai(question)
    if test_all or args.claude:
        test_claude(question)
    if test_all or args.gemini:
        test_gemini(question)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()