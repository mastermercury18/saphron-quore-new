import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
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

sys.stdout = Tee("session_log2.txt")
sys.stderr = sys.stdout  # Redirect errors too


# --- Math Questions Only ---
QUESTION_BANK = [
    {"question": "What is 6 + 3?", "options": ["6", "9", "12"], "answer": 1, "topic": 0},       # basic addition
    {"question": "What is 23 + 18?", "options": ["41", "35", "31"], "answer": 0, "topic": 0},
    {"question": "What is 12 - 5?", "options": ["6", "7", "8"], "answer": 1, "topic": 1},      # subtraction
    {"question": "What is 63 - 12", "options": ["34", "50", "51"], "answer": 2, "topic": 1},     # multiplication
    {"question": "What is 3 * 2", "options": ["6", "5", "4"], "answer": 0, "topic": 2},      # division
    {"question": "What is (2 + 3) * 2?", "options": ["10", "8", "12"], "answer": 2, "topic": 2} # multi-step
]

NUM_TOPICS = 3  # addition/subtraction, multiplication/division, multi-step
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
    
    # If epsilon is large enough, then a random question is picked from the question set 
    # Otherwise, generate the Q-values list and choose the one with highest reward 
    # The question associated with the highest reward provides the most potential for learning 
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0]) #argmax provides index of the greatest Q-value in the list

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
user_knowledge = np.array([0.2, 0.2, 0.2])  # start weak in all topics

# --- Interactive Loop ---

for episode in range(10):
    print(f"\n📘 Quiz Session {episode+1}")
    state = np.copy(user_knowledge)
    
    action = agent.act(state)
    q = QUESTION_BANK[action]
    
    print(f"🧠 Q: {q['question']}")
    for i, option in enumerate(q['options']):
        print(f"  {i}: {option}")
    
    try:
        user_input = int(input("Your answer (0/1/2): ").strip())
    except:
        user_input = -1

    correct = int(user_input == q['answer'])
    print("✅ Correct!" if correct else f"❌ Incorrect. Correct answer is: {q['options'][q['answer']]}")
    
    # Learning update
    topic = q['topic']
    reward = 1 if correct else 0
    if correct:
        user_knowledge[topic] += 0.1  # simulate improvement
    
    next_state = np.copy(user_knowledge)
    agent.remember(state, action, reward, next_state, done=False)
    agent.replay()

    print(f"🧠 Current knowledge: {np.round(user_knowledge, 2)}")

print("\n🎓 Done! The agent should now recommend more of what you need.")

