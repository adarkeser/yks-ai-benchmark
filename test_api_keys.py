#!/usr/bin/env python3
"""Simple test script to verify API keys are working."""

import sys
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("Testing API Keys...\n")
print("=" * 60)

# Test OpenAI
print("\n1. Testing OpenAI API...")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("   ❌ OPENAI_API_KEY not found in .env")
else:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'API key works!'"}],
            max_tokens=10
        )
        print(f"   ✅ OpenAI API works! Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"   ❌ OpenAI API failed: {e}")

# Test Claude (Anthropic)
print("\n2. Testing Claude API...")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("   ❌ ANTHROPIC_API_KEY not found in .env")
else:
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'API key works!'"}]
        )
        print(f"   ✅ Claude API works! Response: {response.content[0].text}")
    except Exception as e:
        print(f"   ❌ Claude API failed: {e}")

# Test Gemini (Google)
print("\n3. Testing Gemini API...")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("   ❌ GOOGLE_API_KEY not found in .env")
else:
    try:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents="Say 'API key works!'"
        )
        print(f"   ✅ Gemini API works! Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Gemini API failed: {e}")

print("\n" + "=" * 60)
print("\nTest complete! If all three show ✅, you're ready to run the benchmark.")

