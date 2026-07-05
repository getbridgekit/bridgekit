from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import parse_provider, get_default_model, Provider
from .providers import create_message

SUPPORTED_TOOLS = ["evaluate", "plan", "redteam"]

SYNTHESIS_PROMPT = """You are comparing two AI-generated outputs for the same analysis task.

Your job is to write a short, direct summary (4-6 sentences) covering:
- Where both outputs agreed
- Where they differed — including any cases where one rated a dimension more harshly than the other
- Which output gave more specific or actionable feedback, and why

Be concrete. Reference specific dimensions or findings. No fluff."""


def _get_tool_fn(tool_name: str):
    if tool_name == "evaluate":
        from .reviewer import evaluate
        return evaluate
    elif tool_name == "plan":
        from .planner import plan
        return plan
    elif tool_name == "redteam":
        from .redteam import redteam
        return redteam
    raise ValueError(f"Unknown tool: {tool_name!r}. Supported tools: {SUPPORTED_TOOLS}")


def _call_tool(tool_name: str, text: str, provider: str, model, kwargs: dict) -> str:
    fn = _get_tool_fn(tool_name)
    if tool_name == "plan":
        return fn(question=text, provider=provider, model=model, **kwargs)
    return fn(text=text, provider=provider, model=model, **kwargs)


def _synthesize(results: list) -> str:
    (provider_a, model_a, output_a), (provider_b, model_b, output_b) = results
    user_message = (
        f"OUTPUT 1 ({provider_a.upper()} / {model_a}):\n{output_a}\n\n"
        f"OUTPUT 2 ({provider_b.upper()} / {model_b}):\n{output_b}"
    )
    return create_message(
        provider=Provider.ANTHROPIC,
        system_prompt=SYNTHESIS_PROMPT,
        user_message=user_message,
        max_tokens=512,
    )


def _format_output(tool: str, summary: str, results: list) -> str:
    divider = "─" * 41
    lines = [
        f"BRIDGEKIT COMPARE: {tool.upper()}",
        divider,
        "",
        "SUMMARY",
        divider,
        summary,
        "",
        "",
    ]
    for i, (provider_name, model, output) in enumerate(results):
        lines.append(f"{provider_name.upper()}  {model}")
        lines.append(divider)
        lines.append(output)
        if i < len(results) - 1:
            lines.append("")
            lines.append("")
    return "\n".join(lines)


def compare(
    text: str,
    tool: str = "evaluate",
    providers: list = None,
    model_a: str = None,
    model_b: str = None,
    **kwargs
) -> str:
    """
    Run the same tool through two providers and return both outputs with a summary.

    Args:
        text: The input text or question to analyze.
        tool: Which tool to run: "evaluate", "plan", or "redteam". Defaults to "evaluate".
        providers: List of exactly two providers to compare. Defaults to ["anthropic", "openai"].
        model_a: Optional model override for the first provider.
        model_b: Optional model override for the second provider.
        **kwargs: Additional arguments passed through to the underlying tool
                  (e.g. max_tokens, stakeholder for redteam, data_description/goal for plan).

    Returns:
        A summary of key differences followed by both full outputs with provider headers.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    if tool not in SUPPORTED_TOOLS:
        raise ValueError(f"Unknown tool: {tool!r}. Supported tools: {SUPPORTED_TOOLS}")

    if providers is None:
        providers = ["anthropic", "openai"]

    if len(providers) != 2:
        raise ValueError(f"providers must contain exactly 2 providers, got {len(providers)}.")

    models = [model_a, model_b]
    resolved_models = []
    for provider, model in zip(providers, models):
        provider_enum = parse_provider(provider)
        resolved_models.append(model if model else get_default_model(provider_enum))

    results_map = {}

    def run_one(idx):
        output = _call_tool(tool, text, providers[idx], models[idx], kwargs)
        return idx, providers[idx], resolved_models[idx], output

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run_one, i): i for i in range(2)}
        for future in as_completed(futures):
            idx, provider, model, output = future.result()
            results_map[idx] = (provider, model, output)

    results = [results_map[0], results_map[1]]
    summary = _synthesize(results)
    return _format_output(tool, summary, results)
