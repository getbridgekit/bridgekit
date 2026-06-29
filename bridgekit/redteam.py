from .config import DEFAULT_MODEL, parse_provider, get_default_model
from .providers import create_message

DEFAULT_STAKEHOLDER = "a skeptical senior executive with no tolerance for weak methodology, unsupported claims, or vague business impact"

SYSTEM_PROMPT_TEMPLATE = """You are {stakeholder}.

You have just read a data science analysis writeup. You are not here to be helpful — you are here to find every weakness, challenge every assumption, and ask the questions that will expose flaws in the analysis.

Your job is to identify the 3 hardest critiques a skeptical stakeholder would make in a readout. For each critique:
- State the challenge directly and bluntly, as the stakeholder would say it
- Explain why this is a legitimate weak point in the analysis
- State what would need to be true for this critique to be addressed

Format your response exactly like this:

BRIDGEKIT RED TEAM
─────────────────────────────────────────
STAKEHOLDER: {stakeholder_label}

CRITIQUE 1: [short title]
❯ [direct challenge as the stakeholder would say it, in quotes]
WHY IT LANDS: [1-2 sentences on why this is a real weakness]
TO ADDRESS: [what would need to be shown or proven]

CRITIQUE 2: [short title]
❯ [direct challenge, in quotes]
WHY IT LANDS: [1-2 sentences]
TO ADDRESS: [what would need to be shown or proven]

CRITIQUE 3: [short title]
❯ [direct challenge, in quotes]
WHY IT LANDS: [1-2 sentences]
TO ADDRESS: [what would need to be shown or proven]

─────────────────────────────────────────
HARDEST QUESTION TO ANSWER
[The single question the stakeholder would ask that would be hardest to answer on the spot]
"""


def redteam(text: str, stakeholder: str = None, provider: str = None, model: str = None, system_prompt: str = None, max_tokens: int = 1024) -> str:
    """
    Red-team a data science analysis writeup from the perspective of a skeptical stakeholder.

    Args:
        text:          Your analysis writeup as a plain string.
        stakeholder:   Optional. The skeptical stakeholder role (e.g. "VP of Finance",
                       "skeptical board member", "Chief Revenue Officer").
                       Defaults to a generic skeptical senior executive.
        provider:      Optional. The AI provider to use ("anthropic", "openai", "gemini").
                       If not specified, defaults to "anthropic" or infers from model.
        model:         Optional. The specific model to use. If not specified, uses the provider's default.
        system_prompt: Optional. A custom system prompt to fully override the default red team persona.
                       When provided, the stakeholder parameter is ignored.
        max_tokens:    Optional. Maximum tokens in the response. Defaults to 1024.

    Returns:
        The 3-5 hardest critiques the stakeholder would make, plus the single
        hardest question to answer on the spot.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    # Parse provider and determine model
    provider_enum = parse_provider(provider, model)
    if model is None:
        model = get_default_model(provider_enum)

    if system_prompt is None:
        stakeholder_label = stakeholder if stakeholder else "Skeptical Senior Executive"
        stakeholder_desc = stakeholder if stakeholder else DEFAULT_STAKEHOLDER
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            stakeholder=stakeholder_desc,
            stakeholder_label=stakeholder_label
        )

    user_message = f"Red-team this analysis writeup:\n\n{text}"

    return create_message(
        provider=provider_enum,
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        max_tokens=max_tokens
    )
