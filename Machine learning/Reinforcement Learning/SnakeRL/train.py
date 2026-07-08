from environment import SnakeEnvironment
from agent import QLearningAgent

env = SnakeEnvironment()

agent = QLearningAgent()

EPISODES = 5000

for episode in range(EPISODES):

    state = env.reset()

    while not env.done:

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.learn(
            state,
            action,
            reward,
            next_state
        )

        state = next_state

    agent.decay()

    print(
        f"Episode {episode+1} "
        f"Score={env.snake.score} "
        f"Epsilon={agent.epsilon:.3f}"
    )

agent.save()
print("saved")