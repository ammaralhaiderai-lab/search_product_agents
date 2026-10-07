# AI Product Decision Agent

A live-web **AI Product Decision Agent**. You provide a product, its intended use, and your maximum budget. The system researches current Saudi-market listings, compares candidates, checks the evidence, retries when the evidence is insufficient, and produces a final recommendation with sources and direct product links.

The project uses LangChain 1.x `create_agent` and LangGraph's Graph API (`StateGraph`) to build and orchestrate the multi-agent system.

## Streamlit App

1. Enter the product you are looking for.
2. Describe what you will use it for.
3. Set your maximum budget in SAR.
4. Run the agent and wait for the research and verification stages.
5. View the recommended products, supporting evidence, prices, stores, specifications, and direct product links.

## On your own machine

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and paste your API keys:

```env
OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit Community Cloud app.
3. Select this repository and the `main` branch.
4. Set `app.py` as the main file.
5. Add the following secrets in Streamlit Community Cloud:

```toml
OPENROUTER_API_KEY = "your_openrouter_key"
TAVILY_API_KEY = "your_tavily_key"
```

Never commit your `.env` file or API keys to GitHub.

## How to submit

1. Make sure the repository is **public** so the project can be accessed from GitHub.
2. Make sure your latest commit contains the working project files and this `README.md`.
3. Keep your `.env` file out of the repository — it contains your API keys and is already included in `.gitignore`.
4. Tag the academy so the submission can be identified by adding this line at the bottom of `README.md`:

   ```markdown
   Submitted by: Ammar Yasser — academy: @SDAIAAcademy
   ```

5. Your submission is complete when the latest commit contains the working application, backend, and the README line above.

## Architecture

```text
User
  ↓
Requirements Agent
  ↓
Planner Agent
  ↓
Parallel Web Search
  ↓
Research Agent
  ↓
Product Normalizer
  ↓
Comparison Agent
  ↓
Critic Agent
  ├── PASS → Final Agent
  │
  └── FAIL → Retry Planner → Research → Comparison → Critic
```

The system also includes loop detection, tool-call limits, checkpoint memory, token/cost tracking, and evidence validation.

## Structure

```text
AI_Product_Decision_Agent/
├── app.py                         # Streamlit frontend
├── agent_backend.py               # Agentic AI backend
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
├── .env.example                   # Environment variable template
├── .gitignore                     # Keeps secrets and local files out of git
└── .streamlit                     # Streamlit configuration
                                   
```

## Quick reference

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, set:

```toml
OPENROUTER_API_KEY = "your_openrouter_key"
TAVILY_API_KEY = "your_tavily_key"
```

## Submission

Submitted by: Ammar Yasser — academy: @SDAIAAcademy
