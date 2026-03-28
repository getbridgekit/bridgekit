from bridgekit import evaluate

# Example: paste your analysis writeup as a string and call evaluate()

text = """
I analyzed 90 days of user behavior data to understand what drives subscription 
upgrades. Users who engaged with the reporting feature within their first week 
were 3x more likely to upgrade within 30 days. I recommend we prioritize 
onboarding users to reporting as a growth lever.
"""

feedback = evaluate(text)
print(feedback)
