import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


# Функция для создания модели дискриминатора
def build_discriminator(input_shape=(16, 16, 1)):
    model = models.Sequential([
        layers.Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=input_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(256, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Функция для загрузки изображений из директории
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.lower().endswith('.png'):
            img = load_img(img_path, color_mode='grayscale', target_size=(16, 16))
            img = img_to_array(img)
            img = img / 255.0
            images.append(img)
    return np.array(images)


# Функция для аугментации данных и сохранения изображений
def augment_and_save_images(data, output_dir, num_images_to_generate=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    it = datagen.flow(data, batch_size=1)
    for i in range(num_images_to_generate):
        batch = it.next()
        image = batch[0].astype('float32')
        plt.imsave(os.path.join(output_dir, f'image_{i + 1}.png'), image.reshape(16, 16), cmap='gray')


def fdata_print(fake_data):
    num_images_to_display = 5

    for i in range(num_images_to_display):
        plt.subplot(1, num_images_to_display, i + 1)
        plt.imshow(fake_data[i].reshape(16, 16), cmap='gray')
        plt.axis('off')

    plt.show()


def main():
    # Создание дискриминатора
    discriminator = build_discriminator()

    # Загрузка аугментированных данных
    folder_path = 'augmented_images'
    real_data = load_images_from_folder(folder_path)  # Реальные аугментированные изображения

    # Генерация фейковых данных для обучения
    fake_data = np.random.rand(len(real_data), 16, 16, 1)  #Фйековые изображения
    fdata_print(fake_data)
    # Лейблы для реальных и фальшивых данных
    real_labels = np.ones((len(real_data), 1))
    fake_labels = np.zeros((len(real_data), 1))

    # Обучение дискриминатора
    d_loss_real = discriminator.train_on_batch(real_data, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
    print(f"D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}")


if __name__ == "__main__":
    main()
