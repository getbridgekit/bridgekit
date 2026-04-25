from .config import DEFAULT_MODEL, parse_provider, get_default_model
from .providers import create_message

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


def plan(question: str, data_description: str = None, goal: str = None, provider: str = None, model: str = None) -> str:
    """
    Recommend the right analytical approach for your problem.

    Args:
        question:         The analytical question you are trying to answer.
        data_description: Optional. A plain text description of your available data.
        goal:             Optional. The goal of your analysis (e.g. "causal inference",
                          "prediction", "segmentation", "hypothesis testing", "exploration").
        provider:         Optional. The AI provider to use ("anthropic", "openai", "gemini").
                          If not specified, defaults to "anthropic" or infers from model.
        model:            Optional. The specific model to use. If not specified, uses the provider's default.

    Returns:
        A structured analytical plan covering the recommended approach, assumptions,
        common pitfalls, and alternatives.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Parse provider and determine model
    provider_enum = parse_provider(provider, model)
    if model is None:
        model = get_default_model(provider_enum)

    user_message = f"Question: {question}"
    if data_description:
        user_message += f"\n\nData: {data_description}"
    if goal:
        user_message += f"\n\nGoal: {goal}"

    return create_message(
        provider=provider_enum,
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        model=model,
        max_tokens=1024
    )
