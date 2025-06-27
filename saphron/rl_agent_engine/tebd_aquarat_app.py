from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import json
import random
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load question bank
with open("aqua_train.json", "r") as f:
    QUESTION_BANK = json.load(f)

NUM_TOPICS = max(q["topic"] for q in QUESTION_BANK) + 1
ACTION_SIZE = len(QUESTION_BANK)
STATE_SIZE = NUM_TOPICS

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

# DQN Agent
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.model = build_model()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95
        self.gamma = 0.95

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        return int(np.argmax(q_values))

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

@app.route("/", methods=["GET", "POST"])
def index():
    if "knowledge" not in session:
        session["knowledge"] = [0.2] * NUM_TOPICS
        session["prev_action"] = None

    state = np.array(session["knowledge"])
    action = agent.act(state)
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
