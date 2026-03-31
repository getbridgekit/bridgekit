from bridgekit import evaluate, ask

# ── Tool #1: Analysis Reviewer ─────────────────────────────────────────────
# Paste your analysis writeup as a string and call evaluate()

text = """
I analyzed 90 days of user behavior data to understand what drives subscription
upgrades. Users who engaged with the reporting feature within their first week
were 3x more likely to upgrade within 30 days. I recommend we prioritize
onboarding users to reporting as a growth lever.
"""

print(evaluate(text))


# ── Tool #2: Analysis Search ───────────────────────────────────────────────
# Point ask() at a folder of documents and ask a question in plain English
# Supports .txt, .md, .pdf, .docx, .pptx, and .ipynb files

print(ask("what drove churn in Q3?", source="sample_data/"))
