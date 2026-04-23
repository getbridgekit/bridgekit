import anthropic
from .config import DEFAULT_MODEL, require_anthropic_api_key

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

def evaluate(text: str) -> str:
    """
    Evaluate a data science analysis writeup and return structured feedback.

    Args:
        text: Your analysis writeup as a plain string.

    Returns:
        Structured feedback across four dimensions.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    api_key = require_anthropic_api_key()

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Please review this analysis writeup:\n\n{text}"
            }
        ]
    )

    return message.content[0].text
