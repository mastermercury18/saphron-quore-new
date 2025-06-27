import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load session log
file = str(input("Enter the file name here: "))

with open(file, "r") as f:
    log = f.read()


# Extract session data
pattern = r"📘 Quiz Session (\d+).*?Q: (.*?)\n.*?Your answer.*?(✅|❌).*?Current knowledge: \[(.*?)\]"
matches = re.findall(pattern, log, re.DOTALL)

data = []
prev_knowledge = None
for session, question, correct_flag, knowledge in matches:
    session = int(session)
    knowledge_vec = list(map(float, knowledge.split()))
    delta = [0]*len(knowledge_vec) if prev_knowledge is None else [k2 - k1 for k1, k2 in zip(prev_knowledge, knowledge_vec)]
    prev_knowledge = knowledge_vec
    data.append({
        "session": session,
        **{f"delta_topic_{i}": delta[i] for i in range(len(delta))}
    })

df = pd.DataFrame(data)

# Prepare heatmap data
heatmap_data = df.set_index("session").T

# Plot heatmap
plt.figure(figsize=(12, 5))
sns.heatmap(heatmap_data, annot=True, center=0, cmap="coolwarm", linewidths=0.5)
plt.title("Knowledge Delta Heatmap by Session")
plt.xlabel("Session")
plt.ylabel("Topic")
plt.tight_layout()
plt.show()
