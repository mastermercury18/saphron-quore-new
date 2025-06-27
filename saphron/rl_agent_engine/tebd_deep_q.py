import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import quimb as qu
import quimb.tensor as qtn
import sys

class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Tee("session_log9.txt")
sys.stderr = sys.stdout  # Redirect errors too


# --- Math Questions Only ---
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

    {"question": "What is 9 / 3?", "options": ["4", "3", "2"], "answer": 1, "topic": 3, "difficulty": 0},
    {"question": "What is 48 / 4?", "options": ["12", "13", "11"], "answer": 0, "topic": 3, "difficulty": 1},
    {"question": "What is 144 / 12?", "options": ["13", "12", "11"], "answer": 1, "topic": 3, "difficulty": 2},

    {"question": "If you buy 3 pens at $2 each, how much?", "options": ["$7", "$5", "$6"], "answer": 2, "topic": 4, "difficulty": 1},
    {"question": "A train travels 60 miles in 2 hours. Speed?", "options": ["40 mph", "30 mph", "50 mph"], "answer": 1, "topic": 4, "difficulty": 2},
    {"question": "You have 5 apples. You eat 2. How many left?", "options": ["3", "2", "4"], "answer": 0, "topic": 4, "difficulty": 0},

    {"question": "What is 1/2 + 1/4?", "options": ["3/4", "1/2", "2/4"], "answer": 0, "topic": 5, "difficulty": 1},
    {"question": "What is 3/5 - 1/5?", "options": ["2/5", "1/5", "3/5"], "answer": 0, "topic": 5, "difficulty": 1},
    {"question": "What is 0.5 + 0.25?", "options": ["1.0", "0.75", "0.5"], "answer": 1, "topic": 6, "difficulty": 1},
    {"question": "What is 1.2 - 0.7?", "options": ["0.4", "0.6", "0.5"], "answer": 2, "topic": 6, "difficulty": 1},

    {"question": "What is 9 + 3?", "options": ["12", "13", "11"], "answer": 0, "topic": 0, "difficulty": 0},
    {"question": "What is 2 * (3 + 5)?", "options": ["16", "14", "12"], "answer": 0, "topic": 4, "difficulty": 2},
    {"question": "If you split $10 between 2 people?", "options": ["$4", "$5", "$6"], "answer": 1, "topic": 4, "difficulty": 0},
]


NUM_TOPICS = 7 
ACTION_SIZE = len(QUESTION_BANK)
STATE_SIZE = NUM_TOPICS

# --- Q-Network ---

# INPUT: [x, y, z]
#   x = mastery level in Topic 0 (e.g., addition)
#   y = mastery level in Topic 1 (e.g., subtraction)
#   z = mastery level in Topic 2 (e.g., multiplication)

# Each value is between 0 and 1, with higher values = better mastery.

# OUTPUT: [a, b, c, d, e, f]
# Each value is the predicted Q-value (expected future reward) for asking a specific question:
#   a = Topic 0, Easy Question
#   b = Topic 0, Hard Question
#   c = Topic 1, Easy Question
#   d = Topic 1, Hard Question
#   e = Topic 2, Easy Question
#   f = Topic 2, Hard Question

# These Q-values are used to select the best action (question to prompt) via argmax
# They represent expected learning gain
def build_model():
    model = tf.keras.Sequential([
        layers.Dense(32, input_dim=STATE_SIZE, activation='relu'),
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

    # Normalize Q-values into probabilities
    probs = np.exp(q_values - np.max(q_values))
    probs /= np.sum(probs)

    # Build the initial biased MPS using qtn.MPS_tensor directly
    initial_states = [np.array([np.sqrt(1 - p), np.sqrt(p)]) for p in probs]
    mps = qtn.COPY_tensor(initial_states, inds='I')

    # TEBD evolution
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

    # Measure
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
    # 
    def __init__(self):
        # Stores past experiences as (state, action, reward, next_state, done)
        # Used for experience replay to stabilize training
        self.memory = deque(maxlen=2000)  
        
        # Deep neural network that predicts Q-values for all possible questions
        # Input: current mastery state → Output: Q-values for each question
        self.model = build_model()  
        
        # Exploration rate — initially 100%, meaning the agent picks random questions
        self.epsilon = 1.0  
        
        # Minimum exploration rate — ensures at least 5% of questions remain randomized
        self.epsilon_min = 0.05  
        
        # After each training round: epsilon *= epsilon_decay
        # Gradually reduces randomness, so agent relies more on learned Q-values over time
        self.epsilon_decay = 0.95  
        
        # Discount factor for future rewards in Q-learning
        # Q(s, a) = reward + gamma * max(Q(next_state))
        # - Small gamma → prioritize immediate learning gain (reward)
        # - Large gamma → prioritize long-term learning gain (future mastery)
        self.gamma = 0.95  
    
    # INPUT: [x, y, z]
    # OUTPUT: index of the question
    
    # Generate the Q-values list and choose the one with highest reward 
    # Use the entanglement-correlations and structural encoding of TEBD to choose the most efficient actions
    def act(self, state, prev_action=None):
        # Exploration
        if np.random.rand() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        
        # Exploitation: structured TEBD selector
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]

        # Correct topic and difficulty info from QUESTION_BANK
        topics = [q['topic'] for q in QUESTION_BANK]
        difficulties = [q.get('difficulty', 0) for q in QUESTION_BANK]  # Example placeholder logic

        # Use TEBD structured selector
        return tebd_structured_action_selector(q_values, topics, difficulties, prev_action=prev_action)


    # INPUT: state, action, reward, next_state, done 
        # state: input vector [x, y, z] of mastery before the question
        # action: index of question currently being prompted to student 
        # reward: 0 or 1, whether the student was correct/incorrect 
        # next_state: updated input vector [x2, y2, z2] of mastery after the question
        # done: boolean checking if the session is over
    #OUTPUT: appends these parameters in a tuple to the memory queue 
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    
    def replay(self, batch_size=8):
        # Skip training if memory queue doesn't have enough experiences 
        if len(self.memory) < batch_size:
            return
        
        #Choose a random sample of experiences from memory queue 
        minibatch = random.sample(self.memory, batch_size)

        #Loop through each experience in the random sample 
        for state, action, reward, next_state, done in minibatch:
            #Effectively saying, Q(s, a) = reward + gamma * max(Q(next_state))
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])
            
            #Vector of current Q-values for this state 
            target_f = self.model.predict(state[np.newaxis, :], verbose=0)

            #Update the Q-value for the question being prompted currently 
            target_f[0][action] = target

            #Train the model after updating the Q-value vector 
            self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)

        #Decrease randomness rate each iteration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Initialize ---
agent = DQNAgent()
user_knowledge = np.array([0.2] * NUM_TOPICS)

# --- Interactive Loop ---

for episode in range(25):
    print(f"\n📘 Quiz Session {episode+1}")
    state = np.copy(user_knowledge)
    
    action = agent.act(state, prev_action=action if episode > 0 else None)
    q = QUESTION_BANK[action]
    
    print(f"Q: {q['question']}")
    for i, option in enumerate(q['options']):
        print(f"  {i}: {option}")
    
    try:
        user_input = int(input("Your answer (0/1/2): ").strip())
    except:
        user_input = -1

    correct = int(user_input == q['answer'])
    print("Correct!" if correct else f"Incorrect. Correct answer is: {q['options'][q['answer']]}")
    
    # Learning update
    topic = q['topic']
    reward = 1 if correct else 0
    if correct:
        user_knowledge[topic] += 0.1  # simulate improvement
    
    next_state = np.copy(user_knowledge)
    agent.remember(state, action, reward, next_state, done=False)
    agent.replay()

    print(f"Current knowledge: {np.round(user_knowledge, 2)}")

print("\n🎓 Done! The agent should now recommend more of what you need.")

