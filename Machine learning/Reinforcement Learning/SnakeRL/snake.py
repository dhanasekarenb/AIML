import pygame

from settings import *


class Snake:

    def __init__(self):

        self.body = [(10, 10)]

        self.direction = (1, 0)

        self.grow = False

        self.score = 0

    def move(self):

        head_x, head_y = self.body[0]

        dx, dy = self.direction

        new_head = (head_x + dx, head_y + dy)

        self.body.insert(0, new_head)

        if not self.grow:
            self.body.pop()
        else:
            self.grow = False

    def eat(self):

        self.grow = True

        self.score += 1

    def draw(self, screen):

        for x, y in self.body:

            pygame.draw.rect(
                screen,
                GREEN,
                (
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                ),
            )

    def hit_wall(self):

        x, y = self.body[0]

        if x < 0 or x >= COLS:
            return True

        if y < 0 or y >= ROWS:
            return True

        return False

    def hit_self(self):

        return self.body[0] in self.body[1:]
    
    def set_direction(self, action):

        directions = [
            (0, -1),   # UP
            (1, 0),    # RIGHT
            (0, 1),    # DOWN
            (-1, 0)    # LEFT
        ]

        current = directions.index(self.direction)

        if action == 0:
            new = current

        elif action == 1:
            new = (current - 1) % 4

        else:
            new = (current + 1) % 4

        self.direction = directions[new]