import pygame


class Brush:
    def __init__(self, size):
        self.size = size
        self.pixels = []


    def draw(self, mouse_position):
        if mouse_position not in self.pixels:
            self.pixels.append(mouse_position)


    def print_brush(self, window, ):
        for pixel in self.pixels:
            pygame.draw.circle(window, (0, 0, 0), pixel, self.size)
