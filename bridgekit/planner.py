import os
import anthropic

SYSTEM_PROMPT = """You are a senior statistician and data scientist advising a colleague on the right analytical approach for their problem.

Given a question, a description of the available data, and the goal of the analysis, recommend the best analytical approach. Be direct and specific — not a textbook, not a list of every possible method.

Structure your response exactly like this:

BRIDGEKIT ANALYSIS PLAN
─────────────────────────────────────────

RECOMMENDED APPROACH
[Name of the method and one sentence on why it fits this problem]

WHY THIS APPROACH
[2-3 sentences on why this is the right fit given the question, data, and goal]

KEY ASSUMPTIONS
[Bullet list of assumptions this approach requires — flag any that may be violated]

WATCH OUT FOR
[The most common mistake DS make on this type of problem]

ALTERNATIVES
[1-2 alternative approaches and when you'd use them instead]

─────────────────────────────────────────
"""


def plan(question: str, data_description: str = None, goal: str = None) -> str:
    """
    Recommend the right analytical approach for your problem.

    Args:
        question:         The analytical question you are trying to answer.
        data_description: Optional. A plain text description of your available data.
        goal:             Optional. The goal of your analysis (e.g. "causal inference",
                          "prediction", "segmentation", "hypothesis testing", "exploration").

    Returns:
        A structured analytical plan covering the recommended approach, assumptions,
        common pitfalls, and alternatives.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found. Set it with: export ANTHROPIC_API_KEY=your_key_here"
        )

    user_message = f"Question: {question}"
    if data_description:
        user_message += f"\n\nData: {data_description}"
    if goal:
        user_message += f"\n\nGoal: {goal}"

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    return message.content[0].text
