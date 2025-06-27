import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Load and parse session log ---
file = str(input("Enter the file name here: "))

with open(file, "r") as f:
    log = f.read()

# --- Extract relevant data ---
pattern = r"📘 Quiz Session (\d+).*?Q: (.*?)\n.*?Your answer.*?(✅|❌).*?Current knowledge: \[(.*?)\]"
matches = re.findall(pattern, log, re.DOTALL)

# --- Convert to structured data ---
data = []
for session, question, correct_flag, knowledge in matches:
    session = int(session)
    correct = 1 if correct_flag == '✅' else 0
    knowledge_vec = list(map(float, knowledge.split()))
    data.append({
        "session": session,
        "question": question.strip(),
        "correct": correct,
        **{f"topic_{i}": knowledge_vec[i] for i in range(len(knowledge_vec))}
    })

df = pd.DataFrame(data)

# --- Calculate performance metrics ---
df["cumulative_correct"] = df["correct"].cumsum()
df["accuracy"] = df["cumulative_correct"] / df["session"]

# --- Visualize knowledge growth ---
melted_knowledge = df.melt(
    id_vars=["session"], 
    value_vars=[col for col in df.columns if col.startswith("topic_")],
    var_name="topic", 
    value_name="mastery"
)

plt.figure(figsize=(10, 6))
sns.lineplot(data=melted_knowledge, x="session", y="mastery", hue="topic", marker="o")
plt.title("Mastery Progress by Topic")
plt.xlabel("Quiz Session")
plt.ylabel("Mastery Level")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()
