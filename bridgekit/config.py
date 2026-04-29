import os
from enum import Enum
from typing import Optional


class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


# Default models for each provider
DEFAULT_MODELS = {
    Provider.ANTHROPIC: "claude-opus-4-6",
    Provider.OPENAI: "gpt-4o",
    Provider.GEMINI: "gemini-1.5-pro"
}

# Legacy support
DEFAULT_MODEL = DEFAULT_MODELS[Provider.ANTHROPIC]


def require_api_key(provider: Provider = Provider.ANTHROPIC) -> str:
    """Return the API key for the specified provider from the environment, or raise a clear error.

    Each Bridgekit tool calls this before constructing a client so
    users get the same friendly message instead of whatever the SDK surfaces
    when the key is missing.
    """
    if provider == Provider.ANTHROPIC:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not found. Set it with: export ANTHROPIC_API_KEY=your_key_here"
            )
        return api_key
    elif provider == Provider.OPENAI:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found. Set it with: export OPENAI_API_KEY=your_key_here"
            )
        return api_key
    elif provider == Provider.GEMINI:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not found. Set it with: export GOOGLE_API_KEY=your_key_here"
            )
        return api_key
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def require_anthropic_api_key() -> str:
    """Legacy function for backward compatibility."""
    return require_api_key(Provider.ANTHROPIC)


def parse_provider(provider: Optional[str] = None, model: Optional[str] = None) -> Provider:
    """Parse provider from provider string or model name."""
    if provider:
        try:
            return Provider(provider.lower())
        except ValueError:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {[p.value for p in Provider]}")
    
    if model:
        # Infer provider from model name
        if model.startswith("claude"):
            return Provider.ANTHROPIC
        elif model.startswith("gpt"):
            return Provider.OPENAI
        elif model.startswith("gemini"):
            return Provider.GEMINI
    
    # Default to Anthropic for backward compatibility
    return Provider.ANTHROPIC


def get_default_model(provider: Provider) -> str:
    """Get the default model for a provider."""
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS[Provider.ANTHROPIC])
