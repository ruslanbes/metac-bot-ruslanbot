# Copying Forecasting-Tools into Bot Repo

## Overview

Copy `forecasting-tools` folder into bot repo to enable local development without publishing changes. This allows GitHub Actions to work since the code is committed to the repo.

## Implementation Steps

### Step 1: Copy the Folder

```bash
# From bot repo root
cp -r ../forecasting-tools ./forecasting-tools

# Remove .git folder to avoid nested repository
rm -rf ./forecasting-tools/.git
```

### Step 2: Update pyproject.toml

**File**: `pyproject.toml`

**Change dependency from:**
```toml
forecasting-tools = "^0.2.80"
```

**To:**
```toml
forecasting-tools = {path = "./forecasting-tools", develop = true}
```

### Step 3: Update .gitignore (if needed)

**File**: `.gitignore`

Ensure `forecasting-tools/` is NOT ignored (it should be committed).

**If present, remove this line:**
```
forecasting-tools/
```

### Step 4: Reinstall Dependencies

```bash
poetry install
# or
poetry update
```