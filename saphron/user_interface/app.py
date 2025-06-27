from flask import Flask, render_template_string, request, session
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import quimb as qu
import quimb.tensor as qtn

# --- Flask setup ---
app = Flask(__name__)

# --- Question Bank ---
QUESTION_BANK = [
    {"question": "What is 5 + 7?", "options": ["11", "12", "13"], "answer": 1, "topic": 0, "difficulty": 0},
    {"question": "What is 15 + 29?", "options": ["43", "42", "44"], "answer": 2, "topic": 0, "difficulty": 1},
    {"question": "What is 103 + 208?", "options": ["311", "321", "301"], "answer": 0, "topic": 0, "difficulty": 2},
    {"question": "What is 12 - 7?", "options": ["4", "6", "5"], "answer": 2, "topic": 1, "difficulty": 0},
    {"question": "What is 63 - 28?", "options": ["36", "35", "34"], "answer": 1, "topic": 1, "difficulty": 1},
    {"question": "What is 523 - 109?", "options": ["424", "414", "413"], "answer": 1, "topic": 1, "difficulty": 2},
    {"question": "What is 4 * 3?", "options": ["11", "12", "13"], "answer": 1, "topic": 2, "difficulty": 0},
    {"question": "What is 17 * 3?", "options": ["51", "52", "50"], "answer": 0, "topic": 2, "difficulty": 1},
    {"question": "What is 123 * 3?", "options": ["379", "359", "369"], "answer": 2, "topic": 2, "difficulty": 2},
]

NUM_TOPICS = 3
ACTION_SIZE = len(QUESTION_BANK)
STATE_SIZE = NUM_TOPICS

# --- TEBD Action Selector ---
def tebd_structured_action_selector(q_values, topics, difficulties, prev_action=None, bond_dim=4, dt=0.1, steps=3):
    q_values = list(q_values)
    n = len(q_values)
    if n < 2:
        return 0

    probs = np.exp(q_values - np.max(q_values))
    probs /= np.sum(probs)

    initial_states = [np.array([np.sqrt(1 - p), np.sqrt(p)]) for p in probs]
    mps = qtn.MPS_tensor(initial_states, site_tag_id='I')

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

# --- Agent ---
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95
        self.gamma = 0.95

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(32, input_dim=STATE_SIZE, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(ACTION_SIZE, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def act(self, state, prev_action=None):
        if np.random.rand() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        topics = [q['topic'] for q in QUESTION_BANK]
        difficulties = [q['difficulty'] for q in QUESTION_BANK]
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

# --- Initialize global agent and knowledge ---
agent = DQNAgent()
user_knowledge = np.array([0.2] * NUM_TOPICS)
current_action = None
episode = 0

# --- Flask UI ---
question_template = """
<h2>Quiz Session {{ episode }}</h2>
<p>{{ question['question'] }}</p>
<form method="POST">
    {% for i, option in enumerate(question['options']) %}
        <input type="radio" name="answer" value="{{ i }}" required> {{ option }}<br>
    {% endfor %}
    <button type="submit">Submit</button>
</form>
<p>Knowledge levels: {{ knowledge }}</p>
"""

result_template = """
<h2>{{ result }}</h2>
<p>The correct answer was: {{ correct }}</p>
<a href="/">Next Question</a>
<p>Knowledge levels: {{ knowledge }}</p>
"""

@app.route('/', methods=['GET', 'POST'])
def next_question():
    global current_action, episode, user_knowledge

    if request.method == 'POST':
        user_input = int(request.form['answer'])
        q = QUESTION_BANK[current_action]
        correct = int(user_input == q['answer'])
        result_text = "✅ Correct!" if correct else "❌ Incorrect."
        topic = q['topic']
        reward = 1 if correct else 0
        if correct:
            user_knowledge[topic] = min(1.0, user_knowledge[topic] + 0.1)

        next_state = np.copy(user_knowledge)
        agent.remember(session['state'], current_action, reward, next_state, done=False)
        agent.replay()

        return render_template_string(result_template,
                                      result=result_text,
                                      correct=q['options'][q['answer']],
                                      knowledge=np.round(user_knowledge, 2))

    state = np.copy(user_knowledge)
    session['state'] = state
    current_action = agent.act(state)
    q = QUESTION_BANK[current_action]
    episode += 1

    return render_template_string(question_template,
                                  episode=episode,
                                  question=q,
                                  knowledge=np.round(user_knowledge, 2))

# --- Run Flask app ---
if __name__ == '__main__':
    app.run(debug=True)
