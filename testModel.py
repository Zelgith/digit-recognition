import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils.resizeImage import resize_images

dataset = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_test = resize_images(x_test, (56, 56))

model = tf.keras.models.load_model('digit_recognition.keras')

numbers_to_test = 50
predictions = model.predict(x_test[:numbers_to_test])

correct = 0

figure = plt.figure(figsize=(14, 8))
subplot_size = np.sqrt(len(predictions))
if len(predictions) > 182:
    subplot_size = np.sqrt(182)

for i in range(len(predictions)):
    guess = np.argmax(predictions[i])
    real = y_test[i]
    print(f"Guess: {guess}, Real: {real}")
    if guess == real:
        correct += 1
    if i < 182:
        figure.add_subplot(int(np.ceil(subplot_size)), int(np.floor(subplot_size)), i + 1)
        plt.imshow(x_test[i], cmap='gray')
        plt.axis("off")
        plt.title(f"Guess: {guess}")

print("Model Accuracy:", correct / len(predictions))
figure.tight_layout()
plt.get_current_fig_manager().set_window_title("Test Output")
plt.show()
