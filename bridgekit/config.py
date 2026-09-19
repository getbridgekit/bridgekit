import os
from enum import Enum
from typing import Optional


class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


# Default models for each provider
DEFAULT_MODELS = {
    Provider.ANTHROPIC: "claude-opus-4-8",
    Provider.OPENAI: "gpt-4o",
    Provider.GEMINI: "gemini-1.5-pro",
    Provider.OLLAMA: "llama3.2",
    Provider.OPENROUTER: "deepseek/deepseek-v4-flash-0731:free",
}

# Legacy support
DEFAULT_MODEL = DEFAULT_MODELS[Provider.ANTHROPIC]

# Common local model family names served by Ollama. Used to infer the
# provider from a bare model name when no explicit provider is given.
OLLAMA_MODEL_PREFIXES = (
    "llama", "mistral", "mixtral", "gemma", "phi", "qwen",
    "codellama", "vicuna", "deepseek", "tinyllama",
)


def require_api_key(provider: Provider = Provider.ANTHROPIC) -> Optional[str]:
    """Return the API key for the specified provider from the environment, or raise a clear error.

    Each Bridgekit tool calls this before constructing a client so
    users get the same friendly message instead of whatever the SDK surfaces
    when the key is missing.

    Ollama runs locally and doesn't use an API key, so this returns None
    for that provider instead of raising.
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
    elif provider == Provider.OLLAMA:
        return None
    elif provider == Provider.OPENROUTER:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not found. Set it with: export OPENROUTER_API_KEY=your_key_here"
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
        elif "/" in model:
            # OpenRouter model names use a "vendor/model" format,
            # e.g. "deepseek/deepseek-v4-flash-0731:free"
            return Provider.OPENROUTER
        elif model.startswith(OLLAMA_MODEL_PREFIXES):
            return Provider.OLLAMA

    # Default to Anthropic for backward compatibility
    return Provider.ANTHROPIC


def get_default_model(provider: Provider) -> str:
    """Get the default model for a provider."""
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS[Provider.ANTHROPIC])
