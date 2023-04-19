import tensorflow as tf
from tensorflow.keras.layers import Conv3D

class LipMotionModelV1(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(LipMotionModelV1, self).__init__()
        # define all layers in init
        # Layer of Block 1
        # self.input_layer = tf.keras.layers.Input(input_shape)
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv3D(filters=1, kernel_size=(1,5,5), activation="relu", input_shape=input_shape),
            tf.keras.layers.Conv3D(filters=32, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(5,5),data_format='channels_last', return_sequences=True),
            tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(5,5),data_format='channels_last', return_sequences=True),
            tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(5,5),data_format='channels_last', return_sequences=True),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=32, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=16, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=8, kernel_size=(1,5,5), activation="relu"),
            tf.keras.layers.Conv3D(filters=4, kernel_size=(1,4,4), activation="relu"),
            tf.keras.layers.Conv3D(filters=4, kernel_size=(1,3,3), activation="relu"),
        ])

    def call(self, input_tensor, training=False):
        # self.input_layer = tf.keras.layers.Input(input_tensor.shape)
        x = self.block1(input_tensor)
        # transpose the tensor to (None, 4, 7, 7, 25)
        x_transposed = tf.transpose(x, perm=[0, 4, 2, 3, 1])

        # reshape the tensor to (None, 4, 49, 25)
        x_reshaped = tf.reshape(x_transposed, shape=(-1, 4, 49, 25))
        return x_reshaped