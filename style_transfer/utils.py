import tensorflow as tf
import numpy as np
from makiflow.base import MakiLayer, MakiTensor
from skimage.io import imsave
import cv2


def clip_image(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]

  return x_var, y_var


def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


def unpreprocess_vgg(x):
    x[..., 0] = x[..., 0] + 103.939
    x[..., 1] = x[..., 1] + 116.779
    x[..., 2] = x[..., 2] + 123.68

    x = x[..., ::-1]
    return x.astype(np.float32)


def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x.astype(np.float32)


def preprocess_input_vgg(x):
    x = x.astype(np.float32)
    mean = [103.939, 116.779, 123.68]
    x = x[..., ::-1]
    x[..., 0] = x[..., 0] - mean[0]
    x[..., 1] = x[..., 1] - mean[1]
    x[..., 2] = x[..., 2] - mean[2]

    return x.astype(np.float32)


def read_image(path, size=(300,300)):
    img = cv2.imread(path)[:, :, ::-1]

    img = cv2.resize(img, size).astype(np.int32)
    return img


def save_image(path, image):
    imsave(path + 'result.jpg', image)


def resize_image(image, size=(300,300), interpolation=cv2.INTER_CUBIC):
    return cv2.resize(image, size, interpolation=interpolation)


# Function to print loss
def update(l):
    print(f'loss {l}')

def style_loss(y,t):
  return tf.reduce_mean(tf.square(gram_matrix(y) - gram_matrix(t)))

def gram_matrix(img):
  # from (H,W,C) - > (C, H*W)
  X = tf.transpose(img,[2, 0, 1])

  X = tf.reshape(X,shape=(X.shape[0], X.shape[1]*X.shape[2]))

  N = tf.constant(X.shape[0].value * X.shape[1].value, dtype=np.float32)

  # According to the paper, author normalized matrix G using N that you can see below
  #N = tf.square(N) * tf.constant(4,dtype=np.float32)

  G = tf.matmul(X, tf.transpose(X)) / N

  return G


# Original InputLayer in MakiFlow have placeholder, in our case we need Variable
class InputLayer_mod(MakiTensor):
    def __init__(self, input_shape, name, value):
        """
        InputLayer is used to instantiate MakiFlow tensor.
        Parameters
        ----------
        input_shape : list
            Shape of input object.
        name : str
            Name of this layer.
        """
        self.params = []
        self._name = str(name)
        self._input_shape = input_shape
        self.input = tf.Variable(value, tf.float32, shape=input_shape, name=self._name)
        super().__init__(
            data_tensor=self.input,
            parent_layer=self,
            parent_tensor_names=None,
            previous_tensors={},
        )

    def get_shape(self):
        return self._input_shape

    def get_name(self):
        return self._name

    def get_params(self):
        return []

    def get_params_dict(self):
        return {}

    def to_dict(self):
        return {
            "name": self._name,
            "parent_tensor_names": [],
            'type': 'InputLayer',
            'params': {
                'name': self._name,
                'input_shape': self._input_shape
            }
        }


def get_bfgs(loss, in_x, epochs):
  return tf.contrib.opt.ScipyOptimizerInterface(loss,var_list=[in_x.get_data_tensor()],
                                                method='L-BFGS-B',
                                                options={'maxiter': epochs},
                                                var_to_bounds=(-127,127)
  )

def get_sgd():
  opt = tf.train.MomentumOptimizer(1000, momentum=0.0, use_nesterov=False)
  minimize =  opt.minimize(loss, var_list=[in_x.get_data_tensor()])
  ses.run(tf.variables_initializer(opt.variables()))
  return minimize