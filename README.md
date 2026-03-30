# Bridgekit

**AI tools that make you a better data scientist, not a redundant one.**

Data scientists are not being replaced — they're being asked to do more with less context, less time, and more pressure to be right. Bridgekit is a growing suite of small, focused tools that bring AI into your existing workflow to sharpen your thinking, catch your blind spots, and level up your craft.

No new interface to learn. No data leaving your hands. Just better work.

---

## Tool #1: Analysis Reviewer

Write your findings the way you normally would. Bridgekit reads them and gives you the feedback a senior data scientist would — before you walk into the meeting.

```python
from bridgekit import evaluate

text = """
I analyzed 90 days of user behavior data to understand what drives subscription 
upgrades. Users who engaged with the reporting feature within their first week 
were 3x more likely to upgrade within 30 days. I recommend we prioritize 
onboarding users to reporting as a growth lever.
"""

evaluate(text)
```

**Output:**

```
BRIDGEKIT FEEDBACK
─────────────────────────────────────────

✅ LOGIC
Your conclusion follows from the data. The 3x lift is a meaningful signal 
worth acting on.

⚠️  WHAT'S MISSING
- Did you control for user intent? Users who explore reporting features may 
  already be power users likely to upgrade regardless.
- What's the sample size behind the 3x figure?
- Is this correlation or did you establish any causal direction?

🎯 WEAKEST POINT
"I recommend we prioritize onboarding to reporting" is a big leap from an 
observational finding. A senior DS would push back on this in the meeting.

💡 LEVEL UP
Look into selection bias and how to address it — this analysis would be 
significantly stronger with a matched cohort or an experiment to validate 
the finding.

─────────────────────────────────────────
```

---

## Installation

**Standard install:**
```bash
pip install bridgekit
```

**In a virtual environment (recommended for clean setups):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install bridgekit
```

**In a Jupyter notebook:**
```python
!pip install bridgekit
```

Requires an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

---

## Getting Started

**From the terminal:**

```bash
python example.py
```

**From a Jupyter notebook:**

Set your API key before launching Jupyter:

```bash
export ANTHROPIC_API_KEY=your_key_here
jupyter notebook
```

Then in a cell:

```python
from bridgekit import evaluate

text = """
Your analysis writeup goes here.
"""

print(evaluate(text))
```

Paste your writeup as a string and call `evaluate()` — that's it.

---

## Tool #2: Analysis Search

Ask questions across a collection of your past analysis documents. Point it at a folder and get answers grounded in your actual work — no digging through files manually.

Uses a vector database and semantic similarity to find relevant context across your documents — not keyword matching.

Supports `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, and `.ipynb` files.

**From a folder:**
```python
from bridgekit import ask

print(ask("what drove churn in Q3?", source="reports/"))
```

**From raw text:**
```python
from bridgekit import ask

text = """
Q3 churn rose to 4.5%, driven by a product outage in August and a pricing
change in July that increased SMB costs by 12%.
"""

print(ask("what caused the Q3 churn spike?", text=text))
```

**Output** *(based on sample data included in the repo)*:
```
Based on the Q3 2024 Churn Analysis, two primary factors drove the elevated
churn rate of 4.5%:

1. August Product Outage — A 14-hour outage affected 3,800 accounts. Impacted
   accounts churned at 8.1% vs 3.2% for unaffected accounts.

2. July Pricing Change — SMB costs increased by an average of 12%, causing SMB
   churn to spike to 7.2% — the highest single-month figure in the dataset.
```

---

## Why not just use Claude?

You could. But you'd need to know what to ask, how to frame it, and what a good answer looks like. Bridgekit has that baked in — it knows you're a data scientist presenting findings, so it asks the right questions automatically. No prompt engineering required. Just paste your work and run it.

It also lives in your Jupyter notebook, so there's no context switching. You stay in your workflow.

---

## Why a library and not a chatbot?

Because your analysis already lives in a notebook. Bridgekit meets you there. A chatbot asks you to re-explain your work from scratch every time. Bridgekit is one function call at the end of your existing process — consistent, reproducible, and fast.

---

## Is my data safe?

Bridgekit only ever sees text you write yourself — your narrative, your conclusions, your writeup. It never touches your raw data, your DataFrames, or your code. You're sending your own words to an API, the same way you'd paste them into a Google Doc to share with a colleague.

---

## What's next?

Bridgekit is a suite, not a one-off. Two tools are live — more are coming:

- **Statistical approach suggester** — describe your problem in plain English, get the right test and why
- **Stakeholder translator** — turn your technical findings into a narrative a non-technical audience will actually follow
- **Assumption checker** — state your analytical assumptions, get the ones you missed

Each tool is small, focused, and built for the way data scientists actually work.

---

## Contributing

Bridgekit is open source and early. If you're a data scientist and something here would genuinely save you time or make you sharper — open an issue, submit a PR, or just tell me what's missing.

---

## License

MIT
