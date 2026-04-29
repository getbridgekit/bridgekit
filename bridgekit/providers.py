"""Provider client factory for Bridgekit multi-provider support."""

from typing import Union, Dict, Any, List
from .config import Provider, require_api_key, get_default_model


class BaseProviderClient:
    """Base class for provider clients."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def create_message(self, system_prompt: str, user_message: str, model: str, max_tokens: int = 1024) -> str:
        """Create a message with the given parameters."""
        raise NotImplementedError("Subclasses must implement create_message")


class AnthropicClient(BaseProviderClient):
    """Anthropic provider client."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def create_message(self, system_prompt: str, user_message: str, model: str, max_tokens: int = 1024) -> str:
        """Create a message using Anthropic's API."""
        message = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": user_message
            }]
        )
        return message.content[0].text


class OpenAIClient(BaseProviderClient):
    """OpenAI provider client."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        import openai
        self.client = openai.OpenAI(api_key=api_key)
    
    def create_message(self, system_prompt: str, user_message: str, model: str, max_tokens: int = 1024) -> str:
        """Create a message using OpenAI's API."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content


class GeminiClient(BaseProviderClient):
    """Google Gemini provider client."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
    
    def create_message(self, system_prompt: str, user_message: str, model: str, max_tokens: int = 1024) -> str:
        """Create a message using Google's Gemini API."""
        import google.generativeai as genai
        
        # Configure the model with system instruction
        model_instance = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
            )
        )
        
        response = model_instance.generate_content(user_message)
        return response.text


def create_client(provider: Provider, model: str = None) -> BaseProviderClient:
    """Create a client for the specified provider."""
    if model is None:
        model = get_default_model(provider)
    
    api_key = require_api_key(provider)
    
    if provider == Provider.ANTHROPIC:
        return AnthropicClient(api_key)
    elif provider == Provider.OPENAI:
        return OpenAIClient(api_key)
    elif provider == Provider.GEMINI:
        return GeminiClient(api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_message(provider: Provider, system_prompt: str, user_message: str, model: str = None, max_tokens: int = 1024) -> str:
    """Create a message using the specified provider."""
    if model is None:
        model = get_default_model(provider)
    
    client = create_client(provider, model)
    return client.create_message(system_prompt, user_message, model, max_tokens)
