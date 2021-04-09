import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
from tensorflow import keras

import time
from tensorflow.keras.layers import Dense, Conv1D, Reshape, Conv1DTranspose, MaxPool1D, Flatten
from tensorflow.keras.layers import LeakyReLU,ReLU
from tensorflow.keras.layers import LayerNormalization,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tqdm import tqdm
from functools import partial
#from IPython import display
num_inputs = int(sys.argv[2])
num_st = int(sys.argv[3])
z_dim = 20
gen_learning_rate=0.001
disc_learning_rate = 0.0001
GP_WEIGHT = 10.0
epochs = 10000
batch_size = int(sys.argv[4])
sample_epoch = 100
sample_epoch_fake = 1000


def build_generator(z_dim):
    model = Sequential()
    model.add(Reshape((z_dim, 1), input_dim=z_dim))

    model.add(Conv1DTranspose(8, kernel_size=4, strides=2))
    model.add(LayerNormalization())
    model.add(LeakyReLU())

    model.add(Conv1DTranspose(8, kernel_size=6, strides=2))
    model.add(LayerNormalization())
    model.add(LeakyReLU())

    model.add(Conv1DTranspose(8, kernel_size=6, strides=1))
    model.add(LayerNormalization())
    model.add(LeakyReLU())

    model.add(Conv1DTranspose(1, kernel_size=7, strides=1, activation='tanh'))


    model.add(Reshape([num_inputs + num_st]))

    # inputs = tf.keras.Input(shape=(z_dim,))
    # x1 = Dense(16)(inputs)
    # x1 = LayerNormalization()(x1)
    # x1 = LeakyReLU(alpha=0.01)(x1)
    # x2 = Reshape((16, 1))(x1)
    # x3 = Conv1DTranspose(1,kernel_size=10, strides=2)(x2)
    # x3 = LayerNormalization()(x3)
    # x3 = LeakyReLU(alpha=0.01)(x3)
    # x4 = Conv1DTranspose(1,kernel_size=10, strides=2)(x3)
    # x4 = LayerNormalization()(x4)
    # x4 = LeakyReLU(alpha=0.01)(x4)
    # x5 = Conv1DTranspose(1,kernel_size=12, strides=1, activation='tanh')(x4)
    # x5 = Reshape([99])(x5)
    # model = tf.keras.Model(inputs=inputs,
    # outputs=x5)

    return model


def build_discriminator():
    model = Sequential()
    model.add(Reshape((num_inputs + num_st, 1), input_dim=num_inputs + num_st))
    model.add(Conv1D(4, kernel_size=5))
    model.add(LayerNormalization())
    model.add(LeakyReLU())
    # model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(Conv1D(4, kernel_size=5))
    model.add(LayerNormalization())
    model.add(LeakyReLU())
    # model.add(MaxPool1D(pool_size=2, strides=1))

    # model.add(Conv1D(1, kernel_size=16, padding='same'))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU())
    # model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(Flatten())
    # model.add(LayerNormalization())
    # model.add(LeakyReLU())

    # model.add(Dense(16))
    # model.add(LayerNormalization())
    # model.add(LeakyReLU())

    model.add(Dense(1))
    return model

def to_mone_one(x, minx, maxx):
    return 2.0 * (x - minx) / (maxx - minx) - 1.0

# inverse of to_mone_one
def to_val(u, minx, maxx):
    return (maxx - minx) * (u + 1.0) / 2.0 + minx

def load_data(fname):
    train_data = pd.read_csv(fname)
    x = train_data.iloc[:,0:].values
    return x


g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
adv_loss_metrics = tf.metrics.Mean(name='adv_loss')

generator_optimizer = tf.keras.optimizers.RMSprop(gen_learning_rate)      #RMSprop   in oreder to test where the error comes from
discriminator_optimizer = tf.keras.optimizers.RMSprop(disc_learning_rate)


def gradient_penalty(discriminator, real_data, fake_data):
    batchsz = real_data.shape[0]

    real_data = tf.cast(real_data, tf.float32)
    fake_data = tf.cast(fake_data, tf.float32)
    # 每个样本随机采样 t，用于插值
    t = tf.random.uniform([batchsz, 1])
    t = tf.broadcast_to(t, real_data.shape)

    # 在真假数据之间做线性插值
    interplate = t * real_data + (1 - t) * fake_data
    # 在梯度环境中计算D 对插值样本的梯度
    with tf.GradientTape() as tape:
        tape.watch([interplate])  # 加入梯度观察列表
        d_interplote_logits = discriminator(interplate)
    grads = tape.gradient(d_interplote_logits, interplate)

    # 计算每个样本的梯度的范数:[b, h, w] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    # 计算梯度惩罚项
    gp = tf.reduce_mean((gp - 1.) ** 2)

    return gp


@tf.function
def train_step(real_data, n_steps=5):
    z = np.random.normal(0, 1, (batch_size, z_dim))
    # z = np.random.uniform(-1., 1., size=[batch_size, z_dim])
    # z = tf.random.uniform([batch_size, z_dim])
    for i in range(n_steps):
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

            fake_data = generator(z, training=False)
            real_output = discriminator(real_data, training=True)
            fake_output = discriminator(fake_data, training=True)
            # data_tensor= tf.convert_to_tensor(data_numpy)

            d_loss = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)
            adv_loss = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
            gp = gradient_penalty(partial(discriminator, training=True), real_data, fake_data)
            d_loss += gp * GP_WEIGHT

            # if tf.math.is_nan(disc_loss) == False:
            gradients_of_discriminator = d_tape.gradient(d_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            '''
            t=0
            for t in range(number_of_disc_layers):
                y = tf.clip_by_value(discriminator.trainable_weights[t],clip_value_min=-0.05,clip_value_max=0.05,name=None)
                discriminator.trainable_weights[t].assign(y)

            #tf.print("jdskjdskfjks", discriminator.trainable_weights[4])'''

            if i == (n_steps - 1):
                # z = np.random.normal(0, 1, (batch_size, z_dim))
                # z = np.random.uniform(-1., 1., size=[batch_size, z_dim])
                fake_data = generator(z, training=True)
                fake_output = discriminator(fake_data, training=False)
                g_loss = -tf.reduce_mean(fake_output)
                gradients_of_generator = g_tape.gradient(g_loss, generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return d_loss, g_loss, adv_loss


def fake_example():
    z = np.random.normal(0, 1, (10, z_dim))
    fake_data = generator.predict(z)
    # d_real = discriminator.predict(train_data)
    d_fake = discriminator.predict(fake_data)

    fake_data_input1 = to_val(fake_data[:, 0:2], -100.0, 100.0).astype(int)
    fake_data_input2 = to_val(fake_data[:, 2:num_inputs], -200.0, 200.0).astype(int)
    fake_data_input = np.concatenate((fake_data_input1, fake_data_input2), axis=1)
    fake_data_path = to_val(fake_data[:, num_inputs:], 0, 3).astype(int)
    fake_data = np.concatenate((fake_data_input, fake_data_path), axis=1)
    fake_data = ','.join(str(x) for x in fake_data)

    f = open('{}time&.txt'.format(sys.argv[7]), 'a')
    f.write(fake_data)
    f.close()
    #for i in range(1):
        #print([int(x) for x in fake_data[i]], end=', ')
        #print(d_fake[i])


discriminator = build_discriminator()

generator = build_generator(z_dim)
# a pandas dataframe to save the loss information to
losses = pd.DataFrame(columns = ['d_loss', 'g_loss', 'adv_loss'])

X_train = load_data('{}'.format(sys.argv[1]))

X_train_input1 = to_mone_one(X_train[:, 0:2], -100.0, 100.0)
X_train_input2 = to_mone_one(X_train[:, 2:num_inputs], -200.0, 200.0)
X_train_path = to_mone_one(X_train[:, num_inputs:], 0, 3)
X_train_input = np.concatenate((X_train_input1, X_train_input2), axis=1)
train_data = np.concatenate((X_train_input, X_train_path), axis=1)

# X_train = to_mone_one(X_train, -10, 10005000)

time_start = time.time()

for epoch in range(epochs):

    d_loss, g_loss, adv_loss = train_step(train_data, 5)
    losses.loc[len(losses)] = d_loss.numpy(), g_loss.numpy(), adv_loss.numpy()
    # print("losses.d_loss.values", losses.d_loss.values[-1])
    g_loss_metrics(g_loss)
    d_loss_metrics(d_loss)
    adv_loss_metrics(adv_loss)

    if (epoch + 1) % sample_epoch == 0:
        print("Epoch: [{}/{}] | d_loss_metrics: {} | g_loss_metrics: {}| adv_loss_metrics: {}"
              .format(epoch, epochs, d_loss_metrics.result(), g_loss_metrics.result(), adv_loss_metrics.result()))
        g_loss_metrics.reset_states()
        d_loss_metrics.reset_states()
        adv_loss_metrics.reset_states()

    if (epoch + 1) % sample_epoch_fake == 0:
        fake_example()

time_to_train_gan = time.time() - time_start
f = open('{}time&.txt'.format(sys.argv[7]), 'a')
f.write('Time for the training is {} sec,'.format(time_to_train_gan))
f.close()

fig = plt.figure(figsize=(20, 12))
plt.ylabel("Loss", fontsize=14, rotation=90)
plt.xlabel("epochs", fontsize=14)
plt.plot(losses.d_loss.values, label='d_loss')
plt.plot(losses.g_loss.values, label='g_loss')
plt.plot(losses.adv_loss.values, label='adv_loss')
plt.legend()
plt.grid(True, which="both")
plt.savefig('{}loss{}.png'.format(sys.argv[7], sys.argv[6]))


z = np.random.normal(0, 1, (100, z_dim))
fake_data = generator.predict(z)
d_fake = discriminator.predict(fake_data)

input1 = to_val(fake_data[:, 0:2], minx=-100.0, maxx=100.0)
input2 = to_val(fake_data[:, 2:num_inputs], minx=-200.0, maxx=200.0)
inputs = np.concatenate((input1,input2),axis=1)
path = to_val(fake_data[:, num_inputs:], minx=0, maxx=3).astype(int)
path = path.astype(float)
fake_data_GAN = np.concatenate((inputs,path),axis=1)

np.savetxt('{}'.format(sys.argv[5]), fake_data_GAN, delimiter = ',')
