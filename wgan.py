import tensorflow as tf
from keras import layers, models, optimizers
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Configuration
IMG_HEIGHT = 22
IMG_WIDTH = 112
CHANNELS = 1
LATENT_DIM = 100
BATCH_SIZE = 64
GP_WEIGHT = 10.0
CRITIC_STEPS = 5
LEARNING_RATE = 0.0002


# Build the Generator
def build_generator():
    """model = keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(128 * 7 * 14),
        layers.Reshape((7, 14, 128)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(32, (5, 5), strides=(1, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(1, (7, 7), strides=(1, 1), padding='valid'),
        layers.Cropping2D(cropping=((3, 3), (0, 0))),  # Output 22x112
        layers.Activation('tanh')
    ])"""
    model = keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(128 * 11 * 56),  # Directly target half dimensions
        layers.Reshape((11, 56, 128)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(128, (5, 5), strides=(2, 1), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),  # 22x56

        layers.Conv2DTranspose(64, (5, 5), strides=(1, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),  # 22x112

        layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same'),
        layers.Activation('tanh')
    ])
    return model


# Build the Critic (Discriminator)
def build_critic():
    model = keras.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),

        layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)  # Linear activation
    ])
    return model

class WGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, critic_steps=CRITIC_STEPS, gp_weight=GP_WEIGHT):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, dis_opt, gen_opt, dis_loss, gen_loss):
        super(WGAN, self).compile()
        self.dis_opt = dis_opt
        self.gen_opt = gen_opt
        self.dis_loss = dis_loss
        self.gen_loss = gen_loss

    def gradient_penalty(self, batch_size, real_signals, fake_signals):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0., 1.)
        diff = fake_signals - real_signals
        interpolated = real_signals + alpha * diff

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_signals):
        if isinstance(real_signals, tuple):
            real_signals = real_signals[0]
        batch_size = tf.shape(real_signals)[0]
        for i in range(self.critic_steps):
            #latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_signals = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_signals, training=True)
                real_logits = self.discriminator(real_signals, training=True)
                #discriminator loss
                d_cost = self.dis_loss(real_logits, fake_logits)
                gp = self.gradient_penalty(batch_size, real_signals, fake_signals)
                d_loss = d_cost + gp * self.gp_weight
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.dis_opt.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_signals = self.generator(random_latent_vectors, training=True)
            gen_sig_logits = self.discriminator(generated_signals, training=True)
            g_loss = self.gen_loss(gen_sig_logits)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        return {'d_loss': d_loss, 'g_loss': g_loss}

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_sig=6, latent_dim=128):
        self.num_sig = num_sig
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_sig, self.latent_dim))
        generated_signals = self.model.generator(random_latent_vectors)

        for i in range(self.num_sig):
            sig = generated_signals[i].numpy()
            sig = keras.preprocessing.image.array_to_img(sig)
            sig.save('gen_signals/generated_sig_{i}_{epoch}.png'.format(i=i, epoch=epoch))

def make_images(train_sig, epochs):
    train_sig_four_dim = tf.expand_dims(train_sig, axis=-1)
    generator = build_generator()
    print(generator.summary())
    critic = build_critic()
    print(critic.summary())

    """c_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5)
    g_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5)"""
    c_optimizer = optimizers.RMSprop(learning_rate=LEARNING_RATE)
    g_optimizer = optimizers.RMSprop(learning_rate=LEARNING_RATE)

    def discriminator_loss(real_sig, fake_sig):
        real_loss = tf.reduce_mean(real_sig)
        fake_loss = tf.reduce_mean(fake_sig)
        return fake_loss - real_loss

    def generator_loss(fake_sig):
        return -tf.reduce_mean(fake_sig)

    cbk = GANMonitor(num_sig=6, latent_dim=LATENT_DIM)

    wgan = WGAN(discriminator=critic, generator=generator, latent_dim=LATENT_DIM)
    wgan.compile(dis_opt=c_optimizer, gen_opt=g_optimizer, dis_loss=discriminator_loss, gen_loss=generator_loss)
    wgan.fit(train_sig_four_dim, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])
    return wgan
