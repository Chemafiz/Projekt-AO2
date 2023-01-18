import cv2
import pickle

import numpy as np
import pygame
import sys
from pygame.locals import *
from brush import Brush
from window import Menu
from tensorflow.keras.models import load_model

supevised_model = pickle.load(open('supervised_model.pickle', 'rb'))
cnn_model = load_model("CNN_model")
cnn_augmentation_model = load_model("CNN_augmentation_model")


def main():
    menu = Menu((800, 600), (255, 255, 255), 60, "Digits predict")
    menu.reset_button = pygame.image.load(r"reset_button.jpg")
    menu.predict_button = pygame.image.load(r"predict_button.jpg")
    menu.reset_button_pos = (25, 50)
    menu.predict_button_pos = (25, 200)

    pygame.init()
    pygame.font.init()
    pygame.display.set_caption(menu.title)
    clock = pygame.time.Clock()
    window = pygame.display.set_mode(menu.window_size)

    my_font1 = pygame.font.SysFont('Calibri', 25, bold=pygame.font.Font.bold)
    my_font2 = pygame.font.SysFont('Calibri', 20, bold=pygame.font.Font.bold)
    text1 = my_font1.render("PREDICTED", False, (0, 0, 0))
    text2 = my_font1.render("DIGIT:", False, (0, 0, 0))
    prediction_ml = my_font2.render("", False, (0, 0, 0))
    prediction_cnn = my_font2.render("", False, (0, 0, 0))
    prediction_cnn_augmentation = my_font2.render("", False, (0, 0, 0))


    brush = Brush(30)
    state = False
    predict = False

    while True:
        window.fill(menu.background_color)
        mouse_position =  pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and mouse_position[0] > 230:
                state = True
            elif event.type == pygame.MOUSEBUTTONDOWN \
                    and menu.reset_button_pos[0] <= mouse_position[0] <= (menu.reset_button_pos[0] + 150) \
                    and menu.reset_button_pos[1] <= mouse_position[1] <= (menu.reset_button_pos[1] + 100):
                brush.pixels.clear()
                prediction_ml = my_font2.render("", False, (0, 0, 0))
                prediction_cnn = my_font2.render("", False, (0, 0, 0))
                prediction_cnn_augmentation = my_font2.render("", False, (0, 0, 0))
            elif event.type == pygame.MOUSEBUTTONDOWN \
                    and menu.predict_button_pos[0] <= mouse_position[0] <= (menu.predict_button_pos[0] + 150) \
                    and menu.predict_button_pos[1] <= mouse_position[1] <= (menu.predict_button_pos[1] + 100):
                predict = True
            if event.type == pygame.MOUSEBUTTONUP:
                state = False

        if state:
            brush.draw(mouse_position)
        brush.print_brush(window)
        pygame.draw.line(window, (0, 0, 0), (200, 0), (200, 600), 4)
        if predict:
            prediction_ml = digit_recognition_supervised(window, supevised_model, my_font2)
            prediction_cnn = digit_recognition_CNN(window, cnn_model, my_font2)
            prediction_cnn_augmentation = digit_recognition_CNN_augmented(window, cnn_augmentation_model, my_font2)

            predict = False

        pygame.draw.line(window, (0, 0, 0), (200, 0), (200, 600), 4)
        window.blit(menu.reset_button, menu.reset_button_pos)
        window.blit(menu.predict_button, menu.predict_button_pos)
        window.blit(text1, (25, 400))
        window.blit(text2, (60, 425))
        window.blit(prediction_ml, (3, 460))
        window.blit(prediction_cnn, (3, 480))
        window.blit(prediction_cnn_augmentation, (3, 500))

        pygame.display.update()
        clock.tick(menu.fps)


def digit_recognition_supervised(window, loaded_model, my_font):
    rect = pygame.Rect(215, 0, 585, 600)
    sub = window.subsurface(rect)
    pygame.image.save(sub, "screenshot.jpg")

    image = cv2.imread("screenshot.jpg", 0)
    down_points = (28, 28)
    image_resized = cv2.resize(image, down_points, interpolation=cv2.INTER_AREA)
    image_final = abs(image_resized - [255])
    cv2.imwrite("resized.jpg", image_resized)

    image_vector = image_final.reshape(1, -1)
    prediction = loaded_model.predict(image_vector)[0]

    return my_font.render(f"Supervised model:{prediction}", False, (0, 0, 0))


def digit_recognition_CNN(window, loaded_model, my_font):
    rect = pygame.Rect(215, 0, 585, 600)
    sub = window.subsurface(rect)
    pygame.image.save(sub, "screenshot.jpg")

    image = cv2.imread("screenshot.jpg", 0)
    down_points = (28, 28)
    image_resized = cv2.resize(image, down_points, interpolation=cv2.INTER_AREA)
    image_final = abs(image_resized - [255])
    cv2.imwrite("resized.jpg", image_resized)

    image_vector = image_final.reshape(1, -1)
    prediction = loaded_model.predict(image_vector.reshape(-1, 28,28,1))
    prediction = np.argmax(prediction)

    return my_font.render(f'CNN:{prediction}', False, (0, 0, 0))


def digit_recognition_CNN_augmented(window, loaded_model, my_font):
    rect = pygame.Rect(215, 0, 585, 600)
    sub = window.subsurface(rect)
    pygame.image.save(sub, "screenshot.jpg")
    image = cv2.imread("screenshot.jpg", 0)
    down_points = (28, 28)
    image_resized = cv2.resize(image, down_points, interpolation=cv2.INTER_AREA)
    image_final = abs(image_resized - [255])
    cv2.imwrite("resized.jpg", image_resized)

    image_vector = image_final.reshape(1, -1)
    prediction = loaded_model.predict(image_vector.reshape(-1, 28,28,1))
    prediction = np.argmax(prediction)

    return my_font.render(f'Augmentation:{prediction}', False, (0, 0, 0))


if __name__ == "__main__":
    main()


