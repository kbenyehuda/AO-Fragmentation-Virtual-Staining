import tensorflow as tf
import tensorflow.keras.layers as layers
import pickle

def build_generator():
    generator = tf.keras.Sequential()
    generator.add(layers.Dense(units=196, input_dim=100))
    generator.add(layers.Reshape([14, 14, 1]))  # 20,16
    generator.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='linear'))
    generator.add(layers.LeakyReLU())
    generator.add(layers.Conv2DTranspose(128, kernel_size=2, strides=2))  # 28,28
    generator.add(layers.Conv2D(64, kernel_size=4, padding='same', activation='linear'))
    generator.add(layers.LeakyReLU())
    generator.add(layers.Conv2DTranspose(64, kernel_size=2, strides=2))  # 56,56
    generator.add(layers.Conv2D(64, kernel_size=4, padding='same', activation='linear'))
    generator.add(layers.LeakyReLU())
    generator.add(layers.Conv2DTranspose(32, kernel_size=2, strides=2))  # 112,112
    generator.add(layers.Conv2D(3, kernel_size=4, padding='same', activation='tanh'))

    # generator.summary()
    return generator



def build_5_generators():
    gens = [build_generator() for i in range(5)]
    for i in range(5):
      pickle_in = open('g_weights_cl' + str(i+1) +'.pickle',"rb")
      dic = pickle.load(pickle_in)
      weights=dic[list(dic.keys())[0]]
      gens[i].set_weights(weights)
    return gens