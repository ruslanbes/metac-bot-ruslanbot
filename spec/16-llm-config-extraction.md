# Extracting LLM Configuration from main.py

## Overview

Current LLM configuration is hard-coded in `main.py` inside the `RuslanBot` instantiation:

- File: `main.py`
- Lines: ~910–928
- Snippet:
  - `llms={ "default": GeneralLlm(...), "summarizer": "...", "researcher": "...", "parser": "..." }`

Goal: move this configuration into a separate, easily editable file while keeping behavior identical.

## Code Changes

### 1. Add llm_config.py

**File**: `llm_config.py` (at repo root, next to `main.py`)

**Contents (initial):**

```python
from forecasting_tools import GeneralLlm

# Default LLM configuration for RuslanBot.
# This mirrors the current inline config in main.py.
DEFAULT_LLMS: dict[str, object] = {
    "default": GeneralLlm(
        model="openrouter/google/gemini-3-pro-preview",
        temperature=0.3,
        timeout=160,
        allowed_tries=2,
    ),
    "summarizer": "openrouter/google/gemini-3-flash-preview",
    "researcher": "openrouter/perplexity/sonar",
    "parser": "openrouter/google/gemini-3-flash-preview",
}
```

### 2. Import config in main.py

**File**: `main.py`

**Change imports near the top:**

Add:

```python
from llm_config import DEFAULT_LLMS
```

Place it after the existing `from forecasting_tools import (...)` import block.

### 3. Use config in RuslanBot instantiation

**File**: `main.py`

**Replace the inline `llms` dict in `template_bot = RuslanBot(...)`:**

Current:

```python
    template_bot = RuslanBot(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/google/gemini-3-pro-preview",  # Using Gemini 3 Pro via OpenRouter
                temperature=0.3,
                timeout=160,
                allowed_tries=2,
            ),
            "summarizer": "openrouter/google/gemini-3-flash-preview",
            "researcher": "openrouter/perplexity/sonar",
            "parser": "openrouter/google/gemini-3-flash-preview",
        },
    )
```

Replace with:

```python
    template_bot = RuslanBot(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms=DEFAULT_LLMS,
    )
```

### 4. How to change models later

To change models or parameters, edit only `llm_config.py`:

- Update `DEFAULT_LLMS["default"]` for main forecasting model
- Update `DEFAULT_LLMS["researcher"]` for research model
- Update `DEFAULT_LLMS["parser"]` and `DEFAULT_LLMS["summarizer"]` as needed

No changes in `main.py` are required once this refactor is done.

