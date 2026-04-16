# Bridgekit

**AI tools that make you a better data scientist, not a redundant one.**

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

Requires an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
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

`data_description` and `goal` are optional — the more context you provide, the more tailored the recommendation.

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
- **Multi-model support** — use any LLM provider (OpenAI, Gemini, open source models via OpenRouter) instead of being tied to Anthropic

Each tool is small, focused, and built for the way data scientists actually work.

---

## Contributing

Bridgekit is open source and early. If you're a data scientist and something here would genuinely save you time or make you sharper — open an issue, submit a PR, or just tell me what's missing.

---

## License

MIT
