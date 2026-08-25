<p align="center">
  <img src="https://raw.githubusercontent.com/getbridgekit/bridgekit/main/assets/logo.png" alt="Bridgekit" width="200"/>
</p>

# Bridgekit

**AI tools that make you a better data scientist, not a redundant one.**

[![PyPI](https://img.shields.io/pypi/v/bridgekit)](https://pypi.org/project/bridgekit/)
[![Downloads](https://img.shields.io/pypi/dm/bridgekit)](https://pypi.org/project/bridgekit/)
[![License](https://img.shields.io/pypi/l/bridgekit)](https://github.com/getbridgekit/bridgekit/blob/main/LICENSE)

Data scientists are not being replaced — they're being asked to do more with less context, less time, and more pressure to be right. Bridgekit is a growing suite of small, focused tools that bring AI into your existing workflow to sharpen your thinking, catch your blind spots, and level up your craft.

No new interface to learn. Just better work.

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

Requires an API key for your chosen provider:

**Anthropic (default):**
```bash
pip install bridgekit
export ANTHROPIC_API_KEY=your_key_here
```

**OpenAI:**
```bash
pip install bridgekit[openai]
export OPENAI_API_KEY=your_key_here
```

**Google Gemini:**
```bash
pip install bridgekit[gemini]
export GOOGLE_API_KEY=your_key_here
```

---

## Getting Started

Set your API key before launching Jupyter:

```bash
export ANTHROPIC_API_KEY=your_key_here
jupyter notebook
```

Then import whichever tool you need:

```python
from bridgekit import evaluate, plan, ask, redteam
```

**Review a writeup:**
```python
print(evaluate("I analyzed 90 days of user behavior data. Users who engaged with the reporting feature were 3x more likely to upgrade."))
```

**Plan your analytical approach:**
```python
print(plan("Did our onboarding flow reduce churn?"))
```

**Search past reports:**
```python
print(ask("What drove churn in Q3?", source="reports/"))
```

---

## Tool #1: Analysis Reviewer

Write your findings the way you normally would. Bridgekit reads them and gives you the feedback a senior data scientist would — to help you improve your work before you walk into the meeting.

```python
from bridgekit import evaluate

text = """
I analyzed 90 days of user behavior data to understand what drives subscription
upgrades. Users who engaged with the reporting feature within their first week
were 3x more likely to upgrade within 30 days. I recommend we prioritize
onboarding users to reporting as a growth lever.
"""

print(evaluate(text))

# Override for longer analyses
print(evaluate(text, max_tokens=2048))
```

**Output:**

```
BRIDGEKIT ANALYSIS REVIEW
─────────────────────────────────────────

1. CLARITY
✅ STRONG — Clean, concise, and jargon-free. Any stakeholder could read
this and immediately understand the claim and the recommendation.

2. STATISTICAL RIGOR
⚠️ NEEDS WORK — "3x more likely" is a compelling number, but critical
context is missing. How many users are in each group? What's the base
upgrade rate? There's no confidence interval or p-value, so we can't
assess whether this difference is statistically significant or noise.

3. METHODOLOGY
❌ MISSING — This reads as a pure correlation finding, but the
recommendation implies causation. Users who explore reporting in week one
may simply be more motivated or already closer to upgrading. Without
addressing the self-selection problem, this recommendation is not
defensible.

4. BUSINESS IMPACT
⚠️ NEEDS WORK — "Growth lever" is directional, not quantified. Translate
the 3x lift into projected revenue or upgrade volume so leadership can
prioritize this against competing initiatives.

─────────────────────────────────────────
BOTTOM LINE
You must address the correlation-vs-causation gap before presenting —
otherwise you risk recommending an onboarding investment that targets a
symptom of upgrade intent rather than a cause of it.
```

---

## Tool #2: Analysis Search

Ask questions across a collection of your past analysis documents. Point it at a folder and get answers grounded in your actual work — no digging through files manually.

Uses a vector database and semantic similarity to find relevant context across your documents — not keyword matching.

Supports `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, and `.ipynb` files.

> **Note:** The first run will download the MiniLM embedding model (~90MB). This is a one-time download — it gets cached locally for all subsequent calls.

**From a folder:**
```python
from bridgekit import ask

print(ask("what drove churn in Q3?", source="reports/"))

# Override for longer responses
print(ask("what drove churn in Q3?", source="reports/", max_tokens=2048))
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

## Tool #3: Analysis Planner

Describe your analytical problem and get a structured plan for the right approach — before you start the analysis.

Covers the recommended method, why it fits your problem, key assumptions, common pitfalls, and alternatives.

```python
from bridgekit import plan

print(plan(
    question="Does our new onboarding flow increase upgrade rates?",
    data_description="We are running an A/B test with ~1,000 users split between old and new onboarding. Key variables will include upgrade status, time to upgrade, acquisition channel, and plan tier.",
    goal="causal inference"
))
```

`data_description`, `goal`, and `max_tokens` are optional — the more context you provide, the more tailored the recommendation.

**`goal` examples:** `"causal inference"`, `"prediction"`, `"segmentation"`, `"hypothesis testing"`, `"exploration"`

**Output:**
```
BRIDGEKIT ANALYSIS PLAN
─────────────────────────────────────────

RECOMMENDED APPROACH
Two-sample proportion test (z-test or Fisher's exact) for the primary
analysis, since you have a randomized experiment with a binary outcome
and want to estimate the causal effect of the new onboarding flow on
upgrade rates.

WHY THIS APPROACH
Randomization handles confounding, so you don't need regression
adjustment to get an unbiased causal estimate. With 500 per group,
you have reasonable power for detecting meaningful differences (~80%
power for a 7-8 percentage point lift from a 20% baseline).

KEY ASSUMPTIONS
- Randomization was correctly implemented (no selection bias)
- No interference between users
- SUTVA: each user has a single well-defined treatment version
- Outcome measurement is complete (watch for differential dropout)
- Users in both arms had equal opportunity to upgrade

WATCH OUT FOR
Peeking and early stopping — if you're checking results repeatedly
before the experiment concludes, your p-values are invalid. Decide
your sample size and analysis time upfront.

ALTERNATIVES
- Logistic regression with covariates (channel, plan tier): use if you
  discover post-hoc imbalance or want to tighten confidence intervals
- Survival analysis (Cox model): use if time-to-upgrade matters as
  much as whether users upgrade
─────────────────────────────────────────
```

---

## Tool #4: Red Team

Simulate a skeptical stakeholder challenging your work to prepare you for the questions you hope no one asks — but now you're ready if they do.

```python
from bridgekit import redteam

text = """
I analyzed 90 days of user behavior data to understand what drives subscription
upgrades. Users who engaged with the reporting feature within their first week
were 3x more likely to upgrade within 30 days. I recommend we prioritize
onboarding users to reporting as a growth lever.
"""

# Default — skeptical senior executive
print(redteam(text))

# Or specify a stakeholder
print(redteam(text, stakeholder="VP of Engineering"))
print(redteam(text, stakeholder="VP of Marketing"))

# Override for longer responses
print(redteam(text, max_tokens=2048))
```

Same writeup, different attack angles:

**VP of Engineering output:**
```
BRIDGEKIT RED TEAM
─────────────────────────────────────────
STAKEHOLDER: VP of Engineering

CRITIQUE 1: Classic correlation-causation conflation
❯ "You're telling me to re-architect our onboarding flow based on a correlation?
Users who dig into reporting in week one are probably already power users.
You haven't shown me that exposing someone to reporting causes them to upgrade."
WHY IT LANDS: No causal identification strategy — no experiment, no instrumental
variable, no matched cohort analysis.
TO ADDRESS: Run an A/B test where randomized new users get guided into reporting
during onboarding. At minimum, a propensity-score-matched comparison controlling
for user segment and acquisition channel.

CRITIQUE 2: 3x on what base rate?
❯ "If the base upgrade rate is 0.5% and reporting users upgrade at 1.5%, I'm not
re-prioritizing my engineering roadmap for that."
WHY IT LANDS: Relative lift without base rates inflates significance. No way to
evaluate whether this justifies the engineering investment.
TO ADDRESS: Absolute upgrade rates, cohort sizes, estimated incremental revenue,
and rough engineering cost to frame ROI.

CRITIQUE 3: No definition of "engaged with reporting"
❯ "What does engaged actually mean? Clicked once? Built a custom report?
If someone accidentally opened the tab, are they in your 3x cohort?"
WHY IT LANDS: The threshold fundamentally changes the interpretation and
the recommended intervention.
TO ADDRESS: Define exact engagement criteria, show sensitivity analysis
across definitions.

─────────────────────────────────────────
HARDEST QUESTION TO ANSWER
"What specific onboarding action would you implement, and what engagement depth
does it need to produce to replicate the effect — and have you tested whether
you can actually get general users to that depth?"
```

**VP of Marketing output:**
```
BRIDGEKIT RED TEAM
─────────────────────────────────────────
STAKEHOLDER: VP of Marketing

CRITIQUE 1: Correlation masquerading as a growth lever
❯ "You're telling me to restructure onboarding based on a correlation. How do
you know that users who found reporting weren't already power users or
higher-intent buyers who would have upgraded regardless?"
WHY IT LANDS: Classic selection bias. Users who proactively explore an advanced
feature in week one are likely more sophisticated or higher-intent. The 3x lift
could entirely reflect who they already were, not what the feature did to them.
TO ADDRESS: A/B test forcing reporting exposure in onboarding vs. not, or a
propensity-matched cohort controlling for acquisition source, company size,
and plan tier.

CRITIQUE 2: Where's the segmentation by channel and campaign?
❯ "I spent millions driving traffic from different channels last quarter.
Did you even look at where these reporting-engaged users came from? If they're
all from our enterprise webinar funnel, this isn't a product insight —
it's a marketing attribution insight you've mislabeled."
WHY IT LANDS: Marketing mix directly shapes user intent. If reporting-engaged
users came disproportionately from specific campaigns, the real lever might be
acquiring more users like them, not changing onboarding.
TO ADDRESS: Break the 3x uplift down by acquisition channel and campaign.
Show the effect holds within channels, not just across the blended population.

CRITIQUE 3: 90 days is a dangerously thin window
❯ "Was there a product launch, a pricing change, or a big campaign push in
that window? How do I know this finding isn't an artifact of whatever else
was happening in those specific 90 days?"
WHY IT LANDS: 90 days is susceptible to confounding events. Any single-quarter
analysis carries high risk of temporal bias.
TO ADDRESS: Replicate the finding across multiple 90-day windows. Flag any
major product, pricing, or marketing events and show the effect persists
after controlling for them.

─────────────────────────────────────────
HARDEST QUESTION TO ANSWER
"If I run an A/B test tomorrow where we force half of new users through a
reporting-focused onboarding flow, what conversion lift are you personally
willing to commit to — and what's your confidence interval on that estimate?"
```

---

## Tool #5: Compare

Run the same tool through two providers and see both outputs side by side. A synthesis summary at the top highlights where the models agreed, where they differed in severity, and which gave more actionable feedback — so you get the key insight without reading both outputs in full. Useful for evaluating which model works best for your use case — as a one-liner.

```python
from bridgekit import compare

text = """
I analyzed 90 days of user behavior data to understand what drives subscription
upgrades. Users who engaged with the reporting feature within their first week
were 3x more likely to upgrade within 30 days. I recommend we prioritize
onboarding users to reporting as a growth lever.
"""

# Compare evaluate across Anthropic and OpenAI (default)
print(compare(text, tool="evaluate"))

# Compare plan across two providers
print(compare("Did our onboarding flow reduce churn?", tool="plan"))

# Compare redteam with a specific stakeholder
print(compare(text, tool="redteam", stakeholder="VP of Finance"))

# Override models for each provider
print(compare(text, model_a="claude-haiku-4-5-20251001", model_b="gpt-4-turbo"))

# Compare two specific providers
print(compare(text, providers=["anthropic", "gemini"]))
```

**Parameters:**
- `tool` - which tool to run: `"evaluate"`, `"plan"`, or `"redteam"` (defaults to `"evaluate"`)
- `providers` - list of exactly two providers to compare (defaults to `["anthropic", "openai"]`)
- `model_a`, `model_b` - optional model overrides for the first and second provider
- `**kwargs` - additional arguments passed through to the underlying tool (e.g. `max_tokens`, `stakeholder`, `data_description`)

**Output:**
```
BRIDGEKIT COMPARE: EVALUATE
─────────────────────────────────────────

SUMMARY
─────────────────────────────────────────
Both outputs rated Clarity as STRONG. They diverged on severity: Anthropic
rated Statistical Rigor as MISSING (harsher) while OpenAI called it NEEDS WORK.
Anthropic gave more specific feedback — naming the correlation-vs-causation
problem explicitly and suggesting concrete fixes. OpenAI's feedback stayed
more generic. Anthropic's bottom line targets the core analytical flaw;
OpenAI's restates the statistical point only.


ANTHROPIC  claude-opus-4-8
─────────────────────────────────────────
BRIDGEKIT ANALYSIS REVIEW
─────────────────────────────────────────

1. CLARITY
✅ STRONG — Clean and jargon-free.

...

OPENAI  gpt-4o
─────────────────────────────────────────
BRIDGEKIT ANALYSIS REVIEW
─────────────────────────────────────────

1. CLARITY
⚠️  NEEDS WORK — The phrase "engagement feature" needs more context.

...
```

Both providers are called in parallel, so the total wait time is the slower of the two — not the sum. The synthesis summary is a third sequential call made after both outputs are ready.

---

## Tool #6: Summarize

Turn a long analysis writeup, notebook, or report into a short executive summary — the gap between doing the analysis and communicating it to people who won't read the whole thing.

```python
from bridgekit import summarize

text = """
I analyzed 90 days of user behavior data to understand what drives subscription
upgrades. Users who engaged with the reporting feature within their first week
were 3x more likely to upgrade within 30 days. Sample size was 1,200 users
across two acquisition channels, with consistent results in both. I recommend
we prioritize onboarding users to reporting as a growth lever.
"""

# Default — general business audience
print(summarize(text))

# Or specify an audience
print(summarize(text, audience="VP of Marketing"))
print(summarize(text, audience="board"))

# Override for longer summaries
print(summarize(text, max_tokens=2048))
```

**Output:**
```
BRIDGEKIT SUMMARY
─────────────────────────────────────────
AUDIENCE: VP of Marketing

KEY TAKEAWAY
Getting users into the reporting feature in their first week is our strongest
predictor of paid upgrades — users who engage with it are 3x more likely to
convert within 30 days.

WHAT WE FOUND
- Early reporting engagement (week 1) drives 3x higher upgrade rates within 30 days
- Finding holds across both acquisition channels, suggesting it's a genuine
  behavior pattern, not channel-specific
- Analyzed 1,200 users over 90 days with sufficient scale to trust the result

SO WHAT
We should redesign onboarding to get users to reporting faster. This is a
high-confidence growth lever worth testing immediately.

─────────────────────────────────────────
```

`audience`, `provider`, `model`, `system_prompt`, and `max_tokens` are all optional — the more specific the audience, the more tailored the summary.

---

## Multi-Provider Support

Bridgekit now supports multiple AI providers so you're not locked into one API. You can use Anthropic, OpenAI, or Google Gemini models with any tool.

**Using different providers:**

```python
from bridgekit import evaluate, plan, ask, redteam

# Use OpenAI (default model: gpt-4o)
print(evaluate("Your analysis here", provider="openai"))

# Use Google Gemini (default model: gemini-1.5-pro)
print(plan("Your question here", provider="gemini"))

# Use specific model
print(redteam("Your analysis here", model="gpt-4-turbo"))
print(ask("Your question here", source="reports/", model="claude-3-opus-20240229"))
```

**Provider auto-detection:**
Bridgekit automatically detects the provider from model names:
- Models starting with "claude" → Anthropic
- Models starting with "gpt" → OpenAI  
- Models starting with "gemini" → Google Gemini

**Default models by provider:**
- Anthropic: `claude-opus-4-8`
- OpenAI: `gpt-4o`
- Gemini: `gemini-1.5-pro`

All tools support the same `provider` and `model` parameters:
- `evaluate(text, provider=None, model=None, system_prompt=None)`
- `plan(question, provider=None, model=None, ..., system_prompt=None)`
- `ask(question, provider=None, model=None, ..., system_prompt=None)`
- `redteam(text, provider=None, model=None, ..., system_prompt=None)`
- `summarize(text, provider=None, model=None, ..., system_prompt=None)`

---

## Custom System Prompts

Every tool accepts an optional `system_prompt` parameter to override the default persona. Use this to adapt the tone or focus to a specific domain without changing anything else.

```python
from bridgekit import evaluate, plan, ask, redteam, summarize

# Narrow the reviewer to a specific domain
print(evaluate("my analysis", system_prompt="You are a skeptical PhD statistician focused only on methodology"))

# Tailor the planner to a specific industry
print(plan("my question", system_prompt="You are a data scientist specializing in healthcare analytics"))

# Replace the red team persona entirely
print(redteam("my analysis", system_prompt="You are a hostile regulator looking for compliance violations"))

# Change the answering style for ask
print(ask("my question", text="...", system_prompt="You are a financial analyst. Answer only in terms of revenue impact."))

# Replace the summarizer persona entirely
print(summarize("my analysis", system_prompt="You are a data journalist writing a one-paragraph news brief."))
```

When `system_prompt` is not provided, each tool uses its built-in default — existing behavior is unchanged.

---

## Why not just use Claude?

You could. But you'd need to know what to ask, how to frame it, and what a good answer looks like. Bridgekit has that baked in — it knows you're a data scientist presenting findings, so it asks the right questions automatically. No prompt engineering required. Just paste your work and run it.

It also lives in your Jupyter notebook, so there's no context switching. You stay in your workflow.

---

## Why a library and not a chatbot?

Because your analysis already lives in a notebook. Bridgekit meets you there. A chatbot asks you to re-explain your work from scratch every time. Bridgekit is one function call — consistent, reproducible, and fast.

---

## What's next?

Bridgekit is a suite, not a one-off. Four tools are live — more are coming:

- **Stakeholder translator** — turn your technical findings into a narrative a non-technical audience will actually follow
- **Assumption checker** — state your analytical assumptions, get the ones you missed
- **More specialized tools** — focused on specific data science workflows and challenges

Each tool is small, focused, and built for the way data scientists actually work.

---

## Contributing

Bridgekit is open source and early. If you're a data scientist and something here would genuinely save you time or make you sharper — open an issue, submit a PR, or just tell me what's missing.

---

## License

MIT
