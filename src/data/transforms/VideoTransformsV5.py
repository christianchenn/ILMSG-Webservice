from src.utils.video import stretch_contrast_v2
import tensorflow as tf
import cv2


class VideoTransformsV5:
    def __init__(self, p1=2, p2=98, normalize=True):
        self.p1 = p1
        self.p2 = p2
        self.normalize = normalize

    
    def transforms(self):

        def preprocess(x):
            # x = stretch_contrast_v2(x, (self.p1, self.p2))
            x = x / 255
            x = tf.image.resize(tf.expand_dims(x, axis=2), [128,128])
            
            return x

        return preprocess