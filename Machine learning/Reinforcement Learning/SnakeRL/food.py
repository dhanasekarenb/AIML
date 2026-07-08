import random
import pygame

from settings import *


class Food:

    def __init__(self):
        self.position = (0, 0)
        self.spawn()

    def spawn(self):
        x = random.randint(0, COLS - 1)
        y = random.randint(0, ROWS - 1)
        self.position = (x, y)

    def draw(self, screen):
        x, y = self.position

        pygame.draw.rect(
            screen,
            RED,
            (
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            ),
        )