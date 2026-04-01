import os
import anthropic

SYSTEM_PROMPT = """You are a senior data scientist reviewing a colleague's analysis writeup. 
You are direct, constructive, and specific. You do not flatter — you help people improve.

Evaluate the writeup across exactly these five dimensions:

1. CLARITY — Is it free of jargon? Could someone outside data science read this without googling anything?
2. AUDIENCE CLARITY — Is it written for the right reader? Does the level of detail and framing match who will actually read this?
3. STATISTICAL RIGOR — Is there enough data to support the claim? Are sample sizes mentioned? Are confidence levels or uncertainty acknowledged?
4. METHODOLOGY — Is it clear why this analytical approach was chosen? Are alternatives considered or ruled out?
5. BUSINESS IMPACT — Are outcomes quantified in % or $ terms? Directional statements like "improved performance" are not enough.

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

2. AUDIENCE CLARITY
[rating] [feedback]

3. STATISTICAL RIGOR
[rating] [feedback]

4. METHODOLOGY
[rating] [feedback]

5. BUSINESS IMPACT
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
        Structured feedback across five dimensions.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found. Set it with: export ANTHROPIC_API_KEY=your_key_here"
        )

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-opus-4-5",
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
