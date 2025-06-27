from flask import Flask, render_template, request, session
import numpy as np
import json
import random
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import quimb as qu
import quimb.tensor as qtn

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load question bank
with open("aqua_train.json", "r") as f:
    QUESTION_BANK = json.load(f)

NUM_TOPICS = max(q["topic"] for q in QUESTION_BANK) + 1
ACTION_SIZE = len(QUESTION_BANK)
STATE_SIZE = NUM_TOPICS

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

@app.route("/", methods=["GET"])
def index():
    if "knowledge" not in session:
        session["knowledge"] = [0.2] * NUM_TOPICS
        session["prev_action"] = None
    state = np.array(session["knowledge"])
    action = agent.act(state, prev_action=session["prev_action"])
    q = QUESTION_BANK[action]
    session["current_question"] = q
    session["prev_action"] = action
    return render_template("index.html", question=q["question"], options=q["options"], qid=action)

@app.route("/submit", methods=["POST"])
def submit():
    answer = int(request.form["answer"])
    q = session["current_question"]
    correct = int(answer == q["answer"])
    topic = q["topic"]
    state = np.array(session["knowledge"])
    reward = 1 if correct else 0
    if correct:
        session["knowledge"][topic] += 0.1
    next_state = np.array(session["knowledge"])
    agent.remember(state, session["prev_action"], reward, next_state, False)
    agent.replay()
    feedback = "Correct!" if correct else f"Incorrect. Correct answer: {q['options'][q['answer']]}"
    return render_template("feedback.html", feedback=feedback, knowledge=session["knowledge"])

if __name__ == "__main__":
    app.run(debug=True)
