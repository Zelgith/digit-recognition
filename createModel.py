import tensorflow as tf
from utils.resizeImage import resize_images

dataset = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = dataset.load_data()

x_train = resize_images(x_train, (56, 56))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.save('digit_recognition.keras')

print("Created model")
