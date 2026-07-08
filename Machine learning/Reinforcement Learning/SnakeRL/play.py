import pygame

from environment import SnakeEnvironment
from agent import QLearningAgent

env = SnakeEnvironment()

agent = QLearningAgent()

agent.load(r"C:\Users\dhana\OneDrive\Desktop\Learning\AIML\Machine learning\Reinforcement Learning\SnakeRL\q_table.pkl")

# No exploration
agent.epsilon = 0

running = True

state = env.reset()

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    action = agent.choose_action(state)

    next_state, reward, done = env.step(action)

    state = next_state

    env.render()

    if done:

        pygame.time.wait(1000)

        state = env.reset()

pygame.quit()