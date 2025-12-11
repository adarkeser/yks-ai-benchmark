"""Prompt templates for the benchmark."""

SYSTEM_MESSAGE = """You are an expert in Turkish university entrance exam (YKS). Your goal is to give the most accurate, concise final answer. Think deeply to arrive at the correct option, but do NOT reveal your reasoning. Output only the single letter A, B, C, D, or E (nothing else). If the question contains an image, analyze it carefully before answering. If uncertain, choose the most likely option."""

USER_MESSAGE = """Bu soruyu çözebilir misin?"""


def get_system_message():
    """Get the system message."""
    return SYSTEM_MESSAGE


def get_user_message():
    """Get the user message."""
    return USER_MESSAGE

