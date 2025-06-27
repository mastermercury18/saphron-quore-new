import json
import re
import random
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

DIFFICULTY_MAP = {
    'easy': 0,
    'medium': 1,
    'hard': 2
}

def estimate_difficulty(question: str) -> int:
    length = len(question)
    if length < 100:
        return DIFFICULTY_MAP['easy']
    elif length < 200:
        return DIFFICULTY_MAP['medium']
    else:
        return DIFFICULTY_MAP['hard']

def format_option(opt: str) -> str:
    return re.sub(r'^[A-E]\)\s*', '', opt).strip()

def cluster_topics(questions: List[str], k: int = 10) -> List[int]:
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(questions)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    return list(kmeans.fit_predict(X))

def convert_aqua_split_with_clustering(split_name: str, out_file: str, k: int = 10):
    ds = load_dataset('deepmind/aqua_rat', split=split_name)
    raw_questions = [item['question'] for item in ds]
    clustered_topics = cluster_topics(raw_questions, k=k)

    output = []
    for i, item in enumerate(ds):
        qtext = item['question']
        opts = [format_option(o) for o in item['options']]
        correct_label = item['correct'].strip()
        try:
            answer_idx = ['A', 'B', 'C', 'D', 'E'].index(correct_label)
        except ValueError:
            answer_idx = 0
        entry = {
            "question": qtext,
            "options": opts,
            "answer": answer_idx,
            "topic": int(clustered_topics[i]),
            "difficulty": estimate_difficulty(qtext)
        }
        output.append(entry)

    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

class QuestionBank:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.load()

    def load(self):
        with open(self.filepath, 'r') as f:
            self.questions: List[Dict[str, Any]] = json.load(f)

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.questions, f, indent=2)

    def add_question(self, question: str, options: List[str],
                     answer: int, topic: int, difficulty: int):
        entry = {
            "question": question,
            "options": options,
            "answer": answer,
            "topic": topic,
            "difficulty": difficulty
        }
        self.questions.append(entry)
        self.save()

    def get_question(self, idx: int) -> Dict[str, Any]:
        return self.questions[idx]

    def get_random(self, topic: Optional[int]=None,
                   difficulty: Optional[int]=None) -> Dict[str, Any]:
        filtered = self.questions
        if topic is not None:
            filtered = [q for q in filtered if q['topic'] == topic]
        if difficulty is not None:
            filtered = [q for q in filtered if q['difficulty'] == difficulty]
        return random.choice(filtered) if filtered else {}

    def list_topics(self) -> List[int]:
        return sorted({q['topic'] for q in self.questions})

    def list_difficulties(self) -> List[int]:
        return sorted({q['difficulty'] for q in self.questions})

if __name__ == "__main__":
    convert_aqua_split_with_clustering('train', 'aqua_train_new.json', k=10)
    convert_aqua_split_with_clustering('validation', 'aqua_dev_new.json', k=10)
    convert_aqua_split_with_clustering('test', 'aqua_test_new.json', k=10)
    print("✅ AQuA-RAT conversion complete with unsupervised topic clustering!")
