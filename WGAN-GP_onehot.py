import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder

import tensorflow as tf

import os
from tensorflow import keras
import time
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import LeakyReLU,ReLU
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tqdm import tqdm
from functools import partial
#from IPython import display
num_inputs = 3
num_st = 96
num_bucket = 9
z_dim = 100


def build_generator(z_dim):
    inputs = tf.keras.Input(shape=(z_dim,))
    x1 = Dense(32)(inputs)
    x1 = LayerNormalization()(x1)
    x1 = LeakyReLU(alpha=0.01)(x1)
    x2 = Dense(64)(x1)
    x2 = LayerNormalization()(x2)
    x2 = LeakyReLU(alpha=0.01)(x2)
    x3 = Dense((num_st + 1) * num_bucket)(x2)
    x3 = LayerNormalization()(x3)
    x4 = tf.keras.layers.Reshape((num_st + 1, num_bucket))(x3)
    x_path, x_input = tf.split(x4, [num_st, 1], axis=1)
    # x_input = LeakyReLU(alpha=0.01)(x_input)
    # x_input = LeakyReLU(alpha=0.01)(x4[:-1,:])
    x_path = tf.nn.softmax(x_path, axis=1)
    outputs = tf.concat([x_path, x_input], 1)
    model = tf.keras.Model(inputs=inputs,
                           outputs=outputs)
    return model


def build_discriminator():
    inputs = tf.keras.Input(shape=(num_st + 1, num_bucket))
    x1 = Flatten()(inputs)
    x2 = Dense(64)(x1)
    x2 = LayerNormalization()(x2)
    x2 = LeakyReLU(alpha=0.01)(x2)
    x3 = Dense(32)(x2)
    x3 = LayerNormalization()(x3)
    x3 = LeakyReLU(alpha=0.01)(x3)
    outputs = Dense(1)(x3)
    model = tf.keras.Model(inputs=inputs,
                           outputs=outputs)

    return model

# function to map [minx, maxx] into [-1,1]
def to_mone_one(x, minx, maxx):
    return 2.0 * (x - minx) / (maxx - minx) - 1.0

# inverse of to_mone_one
def to_val(u, minx, maxx):
    return (maxx - minx) * (u + 1.0) / 2.0 + minx


def load_data(fname):
    train_data = pd.read_csv(fname)
    inputs = train_data.iloc[:, 0:num_inputs].values
    path = train_data.iloc[:, num_inputs:-1].values
    label = train_data.iloc[:, -1].values
    print(np.shape(path))

    # path_one_hot
    path_onehot = keras.utils.to_categorical(path)
    print(np.shape(path_onehot))
    # print(path_onehot)

    # input_padding

    padInput = np.pad(inputs, ((0, 0), (0, 6)), 'constant')
    # print(padInput)

    a = 0
    X_train = np.ones((100, 97, 9))
    while a < 100:
        X_train[a] = np.append(path_onehot[a], [padInput[a]], axis=0)

        a = a + 1
    print(np.shape(X_train))
    print(X_train)

    # path_one_hot
    # enc = OneHotEncoder(categories= [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    # 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    # 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    # 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    # 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    # 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], sparse=False)
    # path_onehot = list(enc.fit_transform(path1))

    return X_train

gen_learning_rate=0.001
disc_learning_rate = 0.001
GP_WEIGHT = 10.0

epochs = 500000
batch_size = 100
sample_epoch = 100
sample_epoch_fake = 1000

# metrics setting
g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
adv_loss_metrics = tf.metrics.Mean(name='adv_loss')

generator_optimizer = tf.keras.optimizers.RMSprop(gen_learning_rate)      #RMSprop   in oreder to test where the error comes from
discriminator_optimizer = tf.keras.optimizers.RMSprop(disc_learning_rate)


# Gradient Penalty (GP)
def gradient_penalty(discriminator, real_data, fake_data):
    batchsz = real_data.shape[0]

    real_data = tf.cast(real_data, tf.float32)
    fake_data = tf.cast(fake_data, tf.float32)
    # 每个样本随机采样 t，用于插值
    t = tf.random.uniform([batchsz, 1, 1])
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
def train_step(real_data, n_steps=4):
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
    z = np.random.normal(0, 1, (1, z_dim))
    fake_data = generator.predict(z)
    #d_real = discriminator.predict(train_data)
    #d_fake = discriminator.predict(fake_data)
    fake_data_path, fake_data_input = tf.split(fake_data, [num_st, 1], axis=1)
    fake_data_path_label = np.argmax(fake_data_path, axis=2)
    fake_data_input = tf.reshape(fake_data_input, [9])
    fake_data_input = fake_data_input[:3]
    fake_data_path_label = tf.reshape(fake_data_path_label, [96])
    fake_data = np.concatenate((fake_data_input, fake_data_path_label), axis=0)
    print(fake_data)
    #print("d_real:%f,d_fake:%+f" % d_real, d_fake )

discriminator = build_discriminator()

generator = build_generator(z_dim)

discriminator.summary()
generator.summary()

# a pandas dataframe to save the loss information to
losses = pd.DataFrame(columns = ['d_loss', 'g_loss', 'adv_loss'])

% % time

X_train = load_data(
    '/Users/guagou/Desktop/code/test_block/hyperg_1F1GAN/hyperg_1F1Test1/wGAN-GP/F18/one_hot/Hy1F1_gsl_sf_hyperg_1F1_F18.csv')

# X_train = to_mone_one(X_train, -10, 10005000)

start = time.time()

for epoch in range(epochs):

    d_loss, g_loss, adv_loss = train_step(X_train, 2)
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

time_to_train_gan = time.time() - start
tf.print('Time for the training is {} sec,'.format(time.time() - start))

fig = plt.figure(figsize=(20, 12))
plt.ylabel("Loss", fontsize=14, rotation=90)
plt.xlabel("epochs", fontsize=14)
plt.plot(losses.d_loss.values, label='d_loss')
plt.plot(losses.g_loss.values, label='g_loss')
plt.plot(losses.adv_loss.values, label='adv_loss')
plt.legend()
plt.grid(True, which="both")

fig = plt.figure(figsize=(20, 12))
plt.ylabel("Loss", fontsize=14, rotation=90)
plt.xlabel("epochs", fontsize=14)
plt.plot(losses.g_loss.values, label='g_loss')
plt.legend()


#plt.legend(['WGAN g loss'],prop={'size': 14}, loc='upper right');
plt.grid(True, which="both")

z = np.random.normal(0, 1, (100, z_dim))
fake_data = generator.predict(z)

fake_data_path, fake_data_input = tf.split(fake_data, [num_st, 1], axis=1)
fake_data_path_label = np.argmax(fake_data_path, axis=2)
fake_data_input = tf.reshape(fake_data_input, [100, 9])
fake_data_input = fake_data_input[:,:3]
fake_data_path_label = tf.reshape(fake_data_path_label, [100, 96])
fake_data = np.concatenate((fake_data_input, fake_data_path_label), axis=1)
print(fake_data)


np.savetxt('/Users/guagou/Desktop/code/test_block/hyperg_1F1GAN/hyperg_1F1Test1/wGAN-GP/F18/one_hot/gpGAN_Hy_F18_1.csv',fake_data, delimiter = ',')
#np.savetxt('/Users/guagou/Desktop/code/test_block/hyperg_1F1GAN/hyperg_1F1Test1/wGAN-GP/F18/gpGAN_Hy_F18_path10.csv',path, delimiter = ',')

#fake_data_GAN = np.concatenate((inputs,path),axis=1)

#for i in range(10):
    #print([int(x) for x in fake_data_GAN[i]], end=', ')
    #print(d_fake[i])