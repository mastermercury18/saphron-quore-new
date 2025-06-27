import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load session log
file = str(input("Enter the file name here: "))

with open(file, "r") as f:
    log = f.read()

# Extract session data
pattern = r"📘 Quiz Session (\d+).*?Q: (.*?)\n.*?Your answer.*?(✅|❌).*?Current knowledge: \[(.*?)\]"
matches = re.findall(pattern, log, re.DOTALL)

data = []
for session, question, correct_flag, knowledge in matches:
    correct = 1 if correct_flag == '✅' else 0
    knowledge_vec = list(map(float, knowledge.split()))
    data.append({
        "question": question.strip(),
        "correct": correct,
        **{f"topic_{i}": knowledge_vec[i] for i in range(len(knowledge_vec))}
    })

df = pd.DataFrame(data)

# Filter only mistakes
mistakes = df[df["correct"] == 0].copy()

# Apply clustering to the mastery state at the time of mistake
X = mistakes[[col for col in mistakes.columns if col.startswith("topic_")]]
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
mistakes["cluster"] = kmeans.labels_

# Visualize cluster centers
plt.figure(figsize=(8, 5))
sns.heatmap(pd.DataFrame(kmeans.cluster_centers_, columns=X.columns), annot=True, cmap="Reds")
plt.title("Mistake Pattern Clusters (by Topic Mastery)")
plt.xlabel("Topic")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()
