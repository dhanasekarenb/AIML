import pygame

from environment import SnakeEnvironment


env = SnakeEnvironment()

running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    env.step(0)

    print(env.get_state())
    
    env.render()

pygame.quit()