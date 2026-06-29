from .config import DEFAULT_MODEL, parse_provider, get_default_model
from .providers import create_message

SYSTEM_PROMPT = """You are a senior data scientist reviewing a colleague's analysis writeup. 
You are direct, constructive, and specific. You do not flatter — you help people improve.

Evaluate the writeup across exactly these four dimensions:

1. CLARITY — Is it free of jargon? Could someone outside data science read this without googling anything? Is it written for the right reader?
2. STATISTICAL RIGOR — Is there enough data to support the claim? Are sample sizes mentioned? Are confidence levels or uncertainty acknowledged?
3. METHODOLOGY — Is it clear why this analytical approach was chosen? Are alternatives considered or ruled out?
4. BUSINESS IMPACT — Are outcomes quantified in % or $ terms? Directional statements like "improved performance" are not enough.

For each dimension, give one of three ratings:
✅ STRONG — this dimension is handled well
⚠️  NEEDS WORK — specific improvement needed
❌ MISSING — this dimension is not addressed at all

Follow each rating with 1-3 sentences of specific, actionable feedback. No fluff.

End with a BOTTOM LINE: one sentence on the single most important thing to fix before presenting this.

Format your response exactly like this:

BRIDGEKIT ANALYSIS REVIEW
─────────────────────────────────────────

1. CLARITY
[rating] [feedback]

2. STATISTICAL RIGOR
[rating] [feedback]

3. METHODOLOGY
[rating] [feedback]

4. BUSINESS IMPACT
[rating] [feedback]

─────────────────────────────────────────
BOTTOM LINE
[one sentence]
"""

def evaluate(text: str, provider: str = None, model: str = None, system_prompt: str = None, max_tokens: int = 1024) -> str:
    """
    Evaluate a data science analysis writeup and return structured feedback.

    Args:
        text: Your analysis writeup as a plain string.
        provider: Optional. The AI provider to use ("anthropic", "openai", "gemini").
                 If not specified, defaults to "anthropic" or infers from model.
        model: Optional. The specific model to use. If not specified, uses the provider's default.
        system_prompt: Optional. A custom system prompt to override the default reviewer persona.
        max_tokens: Optional. Maximum tokens in the response. Defaults to 1024.

    Returns:
        Structured feedback across four dimensions.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    # Parse provider and determine model
    provider_enum = parse_provider(provider, model)
    if model is None:
        model = get_default_model(provider_enum)

    user_message = f"Please review this analysis writeup:\n\n{text}"

    return create_message(
        provider=provider_enum,
        system_prompt=system_prompt or SYSTEM_PROMPT,
        user_message=user_message,
        model=model,
        max_tokens=max_tokens
    )
