import pygame
import numpy as np
import tensorflow as tf
import os
import threading


def canvas_init():
    canvas = []
    for i in range(ROWS):
        canvas.append([])
        for j in range(COLS):
            canvas[i].append(BG_COLOR)
    return canvas


def draw_canvas(screen, canvas):
    for i, row in enumerate(canvas):
        for j, pixel in enumerate(row):
            pygame.draw.rect(screen, pixel, (j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))


def get_grid_position(mouse_pos):
    x, y = mouse_pos
    row = y // PIXEL_SIZE
    col = x // PIXEL_SIZE
    if row >= ROWS:
        raise IndexError
    return row, col


def guess_number(number):
    expanded_number = tf.expand_dims(number, axis=0)
    model = tf.keras.models.load_model('digit_recognition.keras')
    prediction = model.predict(expanded_number)
    return np.argmax(prediction[0])


def draw_prediction(screen, prediction):
    match prediction:
        case 0:
            img = pygame.image.load(os.path.join('numbers', '0.png'))
        case 1:
            img = pygame.image.load(os.path.join('numbers', '1.png'))
        case 2:
            img = pygame.image.load(os.path.join('numbers', '2.png'))
        case 3:
            img = pygame.image.load(os.path.join('numbers', '3.png'))
        case 4:
            img = pygame.image.load(os.path.join('numbers', '4.png'))
        case 5:
            img = pygame.image.load(os.path.join('numbers', '5.png'))
        case 6:
            img = pygame.image.load(os.path.join('numbers', '6.png'))
        case 7:
            img = pygame.image.load(os.path.join('numbers', '7.png'))
        case 8:
            img = pygame.image.load(os.path.join('numbers', '8.png'))
        case 9:
            img = pygame.image.load(os.path.join('numbers', '9.png'))
        case _:
            img = pygame.image.load(os.path.join('numbers', '0.png'))
    screen.blit(img, (WIDTH // 2 + 1, 0))


def draw(screen, canvas):
    draw_canvas(screen, canvas)
    pygame.draw.line(screen, 'white', (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 1)
    pygame.display.update()


if __name__ == '__main__':
    pygame.init()

    WIDTH, HEIGHT = 1120, 560
    ROWS, COLS = 56, 56
    PIXEL_SIZE = HEIGHT // ROWS
    BG_COLOR = (0, 0, 0)
    PAINT_COLOR = (255, 255, 255)
    FPS = 240

    clock = pygame.time.Clock()
    numbers = np.zeros((56, 56))
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Number Guesser')

    canvas = canvas_init()
    screen.fill(BG_COLOR)
    counter = 0
    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                try:
                    row, col = get_grid_position(pos)
                    canvas[row][col] = PAINT_COLOR
                    numbers[row][col] = 255
                except IndexError:
                    pass
            if pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                try:
                    row, col = get_grid_position(pos)
                    canvas[row][col] = BG_COLOR
                    numbers[row][col] = 0
                except IndexError:
                    pass
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DELETE:
                    canvas = canvas_init()
                    numbers = np.zeros((56, 56))

        if counter % 480 == 0:
            draw_prediction(screen, guess_number(numbers))
        draw(screen, canvas)
        counter += 1

    pygame.quit()
