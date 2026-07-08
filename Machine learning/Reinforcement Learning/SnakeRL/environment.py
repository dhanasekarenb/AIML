import pygame

from snake import Snake
from food import Food
from settings import *


class SnakeEnvironment:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        pygame.display.set_caption("Snake RL")

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(None, 30)

        self.reset()

    def reset(self):
        
        self.steps = 0

        self.snake = Snake()

        self.food = Food()

        self.done = False

        self.reward = 0

        return self.get_state()

    def get_state(self):

        head_x, head_y = self.snake.body[0]

        dir_left = self.snake.direction == (-1, 0)
        dir_right = self.snake.direction == (1, 0)
        dir_up = self.snake.direction == (0, -1)
        dir_down = self.snake.direction == (0, 1)

        point_left = (head_x - 1, head_y)
        point_right = (head_x + 1, head_y)
        point_up = (head_x, head_y - 1)
        point_down = (head_x, head_y + 1)

        # Danger Straight
        if dir_right:
            danger_straight = self.is_collision(point_right)
        elif dir_left:
            danger_straight = self.is_collision(point_left)
        elif dir_up:
            danger_straight = self.is_collision(point_up)
        else:
            danger_straight = self.is_collision(point_down)

        # Danger Left
        if dir_right:
            danger_left = self.is_collision(point_up)
        elif dir_left:
            danger_left = self.is_collision(point_down)
        elif dir_up:
            danger_left = self.is_collision(point_left)
        else:
            danger_left = self.is_collision(point_right)

        # Danger Right
        if dir_right:
            danger_right = self.is_collision(point_down)
        elif dir_left:
            danger_right = self.is_collision(point_up)
        elif dir_up:
            danger_right = self.is_collision(point_right)
        else:
            danger_right = self.is_collision(point_left)

        food_x, food_y = self.food.position

        state = [

            danger_straight,
            danger_left,
            danger_right,

            dir_left,
            dir_right,
            dir_up,
            dir_down,

            food_x < head_x,
            food_x > head_x,
            food_y < head_y,
            food_y > head_y

        ]

        return tuple(state)

    def step(self, action):

        self.steps += 1

        # Current distance
        head_x, head_y = self.snake.body[0]
        food_x, food_y = self.food.position

        old_distance = abs(head_x - food_x) + abs(head_y - food_y)

        # Move snake
        self.snake.set_direction(action)
        self.snake.move()

        # New distance
        head_x, head_y = self.snake.body[0]
        new_distance = abs(head_x - food_x) + abs(head_y - food_y)

        # Default reward
        if new_distance < old_distance:
            self.reward = 2
        else:
            self.reward = -2

        # Food
        if self.snake.body[0] == self.food.position:

            self.snake.eat()

            self.food.spawn()

            self.reward = 100

            self.steps = 0 

        # Collision
        if self.snake.hit_wall() or self.snake.hit_self():

            self.reward = -200

            self.done = True

        # Too many steps without eating
        if self.steps > 100 * len(self.snake.body):

            self.reward = -50

            self.done = True

        state = self.get_state()

        return state, self.reward, self.done
    

    def render(self):

        self.screen.fill(BLACK)

        self.food.draw(self.screen)

        self.snake.draw(self.screen)

        score = self.font.render(
            f"Score : {self.snake.score}",
            True,
            WHITE,
        )

        self.screen.blit(score, (10, 10))

        pygame.display.flip()

        self.clock.tick(FPS)


    def is_collision(self, point):

        x, y = point

        if x < 0 or x >= COLS:
            return True

        if y < 0 or y >= ROWS:
            return True

        if point in self.snake.body[1:]:
            return True

        return False