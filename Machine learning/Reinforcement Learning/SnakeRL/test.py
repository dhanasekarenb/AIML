from agent import QLearningAgent

agent = QLearningAgent()

state = (False,) * 11

print(agent.choose_action(state))

print(agent.q_table)