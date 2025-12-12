"""Prompt templates for the benchmark."""

SYSTEM_MESSAGE = """You are an expert in Turkish university entrance exam (YKS). Your goal is to give the most accurate answer with clear reasoning.

CRITICAL: You MUST end your response with the exact format "Answer: X" where X is A, B, C, D, or E.

Instructions:
1. Analyze the question carefully (including any images)
2. Provide VERY BRIEF reasoning (MAXIMUM 3 sentences explaining your thought process)
3. On a new line, write EXACTLY: "Answer: X" where X is ONE letter (A, B, C, D, or E)

Example response format:
"The passage discusses... [your reasoning here].

Answer: C"

IMPORTANT: Your response MUST end with "Answer: X" on its own line."""

USER_MESSAGE = """Bu soruyu çözebilir misin?"""


def get_system_message():
    """Get the system message."""
    return SYSTEM_MESSAGE


def get_user_message():
    """Get the user message."""
    return USER_MESSAGE

