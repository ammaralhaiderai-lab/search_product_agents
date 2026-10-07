# AI Product Decision Agent — Streamlit

This project converts the working Colab notebook into a Streamlit app without changing the core agent architecture.

## Structure

```text
ai_product_streamlit_app/
├── app.py
├── agent_backend.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## Local setup in VS Code

Recommended: use Python 3.12 locally so your environment matches the default Python version currently used by Streamlit Community Cloud.

### 1. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Add your API keys

Copy `.env.example` to `.env` and put your real keys there:

```text
OPENROUTER_API_KEY=...
TAVILY_API_KEY=...
```

Do not commit `.env`.

### 4. Run Streamlit

```powershell
streamlit run app.py
```

The app opens in your browser.

## What the app does

Page 1:
- Product name
- Purpose / use case
- Maximum budget

The app converts those fields into a user request and sends it to the existing agent pipeline.

Page 2:
- Final agent recommendation
- Product candidates
- Price
- Saudi source/store
- Specifications available from the agent
- Evidence
- Direct product link

## Streamlit Community Cloud

The project already includes `requirements.txt`, which is the standard dependency file for Community Cloud.

1. Push these files to a GitHub repository.
2. Create a Streamlit Community Cloud app using `app.py` as the entrypoint.
3. In Advanced settings, use Python 3.12.
4. In Secrets, add:

```toml
OPENROUTER_API_KEY = "..."
TAVILY_API_KEY = "..."
```

Do not upload `.env` or real secret files to GitHub.

## Important note

The agent uses live web research, so price and stock can change. The app shows the source URL so the user can verify the final listing directly.

## Current agent source

`agent_backend.py` is derived from the working notebook `Untitled3 (1)(2).ipynb`. The production changes are limited to:
- replacing Colab-only installation/setup with standard Python environment loading
- exposing the existing async pipeline to Streamlit
- removing notebook test cells from the production module
- adding the Streamlit UI
