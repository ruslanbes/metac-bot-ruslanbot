from forecasting_tools import GeneralLlm

# Default LLM configuration for RuslanBot.
# Edit this file to change models or parameters; no changes needed in main.py.
DEFAULT_LLMS: dict[str, object] = {
    "default": GeneralLlm(
        model="openrouter/google/gemini-3-pro-preview",
        temperature=0.3,
        timeout=160,
        allowed_tries=2,
    ),
    "summarizer": "openrouter/google/gemini-3-flash-preview",
    "researcher": "openrouter/perplexity/sonar",
    #"researcher": "asknews/news-summaries",
    "parser": "openrouter/google/gemini-3-flash-preview",
}
