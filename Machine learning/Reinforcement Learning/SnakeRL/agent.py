import random
import pickle

class QLearningAgent:

    def __init__(self):

        self.q_table = {}

        self.learning_rate = 0.1

        self.discount = 0.9

        self.epsilon = 1.0

        self.epsilon_decay = 0.995

        self.epsilon_min = 0.01

    def get_q_values(self, state):

        if state not in self.q_table:

            self.q_table[state] = [0, 0, 0]

        return self.q_table[state]
    
    def choose_action(self, state):

        q_values = self.get_q_values(state)   # Always initialize the state

        if random.random() < self.epsilon:
            return random.randint(0, 2)

        return q_values.index(max(q_values))
    
    def learn(
    self,
    state,
    action,
    reward,
    next_state,
    ):

        current_q = self.get_q_values(state)[action]

        max_future_q = max(self.get_q_values(next_state))

        new_q = current_q + self.learning_rate * (

            reward +

            self.discount * max_future_q

            - current_q

        )

        self.q_table[state][action] = new_q

    def decay(self):

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

    def save(self, filename="q_table.pkl"):

        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filename="q_table.pkl"):

        try:

            with open(filename, "rb") as f:

                self.q_table = pickle.load(f)

            print("Q-table loaded.")

        except FileNotFoundError:

            print("No saved Q-table found.")