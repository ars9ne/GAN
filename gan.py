import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import time

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

# Функция для создания модели генератора
def build_generator(latent_dim, output_shape=(16, 16, 1)):
    model = models.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((4, 4, 16)),  #старт с 4x4
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),  # повышение до 8x8
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),  # повышение дo 16x16
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(1, kernel_size=4, padding='same', activation='tanh')
    ])
    return model

# Функция для создания и компиляции GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Заморозка дискриминатора во время обучения генератора
    model = models.Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer='adam')
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

# Функция для сохранения сгенерированных изображений
def save_generated_images(generator, latent_dim, epoch, num_images=5, save_dir="epochs"):
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    gen_imgs = generator.predict(noise)

    fig, axs = plt.subplots(1, num_images, figsize=(20, 4))
    for i in range(num_images):
        axs[i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
        axs[i].axis('off')


    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_combined_images.png"))
    plt.close()


# Функция для обучения GAN
def train_gan(generator, discriminator, gan, real_data, epochs, batch_size, latent_dim):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Реальные изображения
        idx = np.random.randint(0, real_data.shape[0], half_batch)
        real_imgs = real_data[idx]

        # Сгенерированные изображения
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_imgs = generator.predict(noise)

        # Тренировка дискриминатора
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Тренировка генератора
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)

        # Вывод прогресса
        print(f"Эпоха {epoch + 1}/{epochs}, D loss: {d_loss}, G loss: {g_loss}")

        # Сохранение образцов из разных эпох
        if epoch in {500, 1000, 2000, 3500, 5000, 9999}:
            save_generated_images(generator, latent_dim, epoch)

# Главная функция
def main():
    latent_dim = 100
    discriminator = build_discriminator()
    generator = build_generator(latent_dim)
    gan = build_gan(generator, discriminator)

    folder_path = 'augmented_images'
    real_data = load_images_from_folder(folder_path)
    train_gan(generator, discriminator, gan, real_data, epochs=10000, batch_size=32, latent_dim=latent_dim)

if __name__ == "__main__":
    main()
