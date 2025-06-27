from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import random
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import os
import quimb as qu
import quimb.tensor as qtn

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load question bank
with open("aqua_train_new.json", "r") as f:
    QUESTION_BANK = json.load(f)

# Determine all unique topics from the question bank
unique_topics = sorted(set(q["topic"] for q in QUESTION_BANK))
NUM_TOPICS = len(unique_topics)
ACTION_SIZE = len(QUESTION_BANK)
STATE_SIZE = NUM_TOPICS

print(f"Found {NUM_TOPICS} topics: {unique_topics}")

# Global session storage (in production, use Redis or database)
user_sessions = {}

# Build model
def build_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(STATE_SIZE,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(ACTION_SIZE, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
    return model

# TEBD structured action selector
def tebd_structured_action_selector(q_values, topics, difficulties, prev_action=None, bond_dim=4, dt=0.1, steps=3):
    q_values = list(q_values)
    n = len(q_values)
    if n < 2:
        return 0
    probs = np.exp(q_values - np.max(q_values))
    probs /= np.sum(probs)
    initial_states = [np.array([np.sqrt(1 - p), np.sqrt(p)]) for p in probs]
    mps = qtn.COPY_tensor(initial_states, inds='I')
    for _ in range(steps):
        for i in range(n - 1):
            t1, d1 = topics[i], difficulties[i]
            t2, d2 = topics[i + 1], difficulties[i + 1]
            penalty = 0.0
            if prev_action in [i, i + 1]:
                penalty += 0.5
            if t1 == t2:
                penalty += 0.25
            if d1 == d2:
                penalty += 0.25
            coupling = (q_values[i] * q_values[i + 1]) - penalty
            H = np.kron(np.eye(4), np.eye(4)) - dt * coupling * np.eye(16)
            U = qu.expm(-H * dt).reshape(2, 2, 2, 2)
            mps.apply_two_site_gate(U, i, i + 1, contract='swap+split', max_bond=bond_dim)
    final_probs = []
    for i in range(n):
        red = mps.partial_trace_complement(i)
        z_expect = red.expectation(qu.pauli('Z'))
        prob_1 = (1 - z_expect) / 2
        final_probs.append(max(0, prob_1))
    final_probs = np.array(final_probs)
    final_probs /= np.sum(final_probs)
    return int(np.random.choice(n, p=final_probs))

# DQN Agent
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.model = build_model()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95
        self.gamma = 0.95

    def act(self, state, prev_action=None):
        if np.random.rand() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        topics = [q['topic'] for q in QUESTION_BANK]
        difficulties = [q.get('difficulty', 0) for q in QUESTION_BANK]
        return tebd_structured_action_selector(q_values, topics, difficulties, prev_action=prev_action)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=8):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis, :], verbose=0)
            target_f[0][action] = target
            self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

agent = DQNAgent()

def get_or_create_session(session_id):
    """Get or create a user session"""
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "knowledge": [0.1] * NUM_TOPICS,
            "prev_action": None,
            "current_question": None
        }
    return user_sessions[session_id]

@app.route("/api/question", methods=["GET"])
def get_question():
    # For simplicity, using a default session ID
    # In production, implement proper session management
    session_id = "default"
    session_data = get_or_create_session(session_id)
    
    state = np.array(session_data["knowledge"])
    action = agent.act(state, prev_action=session_data.get("prev_action"))
    q = QUESTION_BANK[action]
    
    session_data["current_question"] = q
    session_data["prev_action"] = action
    
    print(f"Current knowledge state: {session_data['knowledge']}")
    print(f"Selected question topic: {q['topic']}")
    
    return jsonify({
        "question": q["question"],
        "options": q["options"],
        "qid": action,
        "topic": q["topic"],
        "knowledge": session_data["knowledge"]
    })

@app.route("/api/submit", methods=["POST"])
def submit_answer():
    data = request.get_json()
    answer = int(data["answer"])
    
    session_id = "default"
    session_data = get_or_create_session(session_id)
    
    q = session_data["current_question"]
    correct = int(answer == q["answer"])
    topic = q["topic"]
    
    # Topic is already an integer index (0-9)
    topic_index = topic
    
    print(f"Question topic: {topic}, topic_index: {topic_index}")
    print(f"Knowledge before update: {session_data['knowledge']}")
    
    state = np.array(session_data["knowledge"])
    reward = 1 if correct else 0
    
    if correct:
        session_data["knowledge"][topic_index] += 0.1
        # Ensure knowledge doesn't exceed 1.0
        session_data["knowledge"][topic_index] = min(1.0, session_data["knowledge"][topic_index])
    
    print(f"Knowledge after update: {session_data['knowledge']}")
    
    next_state = np.array(session_data["knowledge"])

    agent.remember(state, session_data["prev_action"], reward, next_state, False)
    agent.replay()

    feedback_text = "Correct!" if correct else f"Incorrect. Correct answer: {q['options'][q['answer']]}"
    
    return jsonify({
        "correct": bool(correct),
        "feedback": feedback_text,
        "knowledge": session_data["knowledge"],
        "topic_scores": list(enumerate(session_data["knowledge"]))
    })

@app.route("/api/reset", methods=["POST"])
def reset_session():
    """Reset user session"""
    session_id = "default"
    if session_id in user_sessions:
        del user_sessions[session_id]
    return jsonify({"message": "Session reset successfully"})

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get learning statistics"""
    session_id = "default"
    session_data = get_or_create_session(session_id)
    
    return jsonify({
        "knowledge": session_data["knowledge"],
        "total_questions": len(QUESTION_BANK),
        "topics": unique_topics,
        "agent_epsilon": agent.epsilon
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001) 