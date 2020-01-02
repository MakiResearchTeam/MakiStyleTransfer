import tensorflow as tf
import numpy as np

from .model_builder import create_model_vgg16
from .utils import read_image, preprocess_input_vgg, unpreprocess_vgg, scale_img, style_loss, get_bfgs, update

class StyleTransferModel:

    def __init__(self, alpha, beta, path_to_weights, path_to_style_img,
                 path_to_content_img, image_size=(300,300), nb_layers=15):
        # alpha - for content, beta - for style, input_shape = (1,300,300,3)
        self._style_img = read_image(path_to_style_img, image_size)
        self._content_img = read_image(path_to_content_img, image_size)
        self._final_img = None

        input_shape = (1, image_size[0], image_size[1], 3)
        self._norm_style_img = preprocess_input_vgg(self._style_img).reshape(input_shape)
        self._norm_content_img = preprocess_input_vgg(self._content_img).reshape(input_shape)

        self._ses = None
        self._opt = None
        self._use_bfgs = None

        self.__update_session()
        self.__count_targets(nb_layers, path_to_weights, input_shape)
        self.__build_final_loss(nb_layers, path_to_weights, input_shape, alpha, beta)

    def __update_session(self):
        if self._ses is not None:
            self._ses.close()
        self._ses = tf.Session()

    def __build_model_and_load_weights(self,nb_layers, path_to_weights, input_shape, type, xinp=None):
        model, in_x, output, names, outputs = create_model_vgg16(input_shape,
                                                                 xinp=xinp,
                                                                 mode=type,
                                                                 number_of_layers=nb_layers
        )

        model.set_session(self._ses)
        model.load_weights(path_to_weights, layer_names=names)

        return model, in_x, output, names, outputs

    def __count_targets(self, nb_layers, path_to_weights, input_shape):
        model, in_x, output, names, outputs = self.__build_model_and_load_weights(nb_layers,
                                                                                  path_to_weights,
                                                                                  input_shape,
                                                                                  'Network'
        )

        self._content_target = [
            self._ses.run(out.get_data_tensor(),
                 feed_dict={in_x.get_data_tensor(): self._norm_content_img}).astype(np.float32) for out in outputs
        ]

        self._style_target = [
            self._ses.run(out.get_data_tensor(),
                feed_dict={in_x.get_data_tensor(): self._norm_style_img}).astype(np.float32) for out in outputs
        ]

        self.__update_session()

    def __build_final_loss(self,nb_layers, path_to_weights, input_shape, alpha, beta):
        x = np.random.randn(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
        
        model, in_x, output, names, outputs = self.__build_model_and_load_weights(nb_layers,
                                                                                  path_to_weights,
                                                                                  input_shape,
                                                                                  'Style',
                                                                                   xinp=x
        )
        self._model = model

        if type(alpha) is list and len(alpha) != len(outputs):
            raise ValueError('number of alpha and output layers have different sizes')

        if type(beta) is list and len(beta) != len(outputs):
            raise ValueError('number of beta and output layers have different sizes')

        self._alpha = tf.constant(alpha, dtype=np.float32)

        self._beta = tf.constant(beta, dtype=np.float32)

        self._content_target = [tf.Variable(target) for target in self._content_target]
        self._style_target = [tf.Variable(target) for target in self._style_target]

        self._loss = None

        # Build content loss
        for i,sym, act in enumerate(zip(outputs,self._content_target)):
            temp_answer = tf.reduce_mean(tf.square(sym.get_data_tensor() - act))

            if type(alpha) is list:
                temp_answer = temp_answer * self._alpha[i]
            else:
                temp_answer = temp_answer * self._alpha

            if self._loss is None:
                self._loss = temp_answer
            else:
                self._loss += temp_answer

        # Build style loss
        for i,sym, act in enumerate(zip(outputs,self._style_target)):
            temp_answer = style_loss(sym.get_data_tensor()[0], act[0])

            if type(beta) is list:
                temp_answer = temp_answer * self._beta[i]
            else:
                temp_answer = temp_answer * self._beta

            self._loss += temp_answer

        # Initialize variables
        self._ses.run(tf.variables_initializer(
            [
                model.get_node('Input').get_data_tensor()
            ] + [
                tar for tar in self._style_target
            ] + [
                tar for tar in self._content_target
            ]
        ))

    def compile_optimizer(self, optimizer='L-BFGS-B'):
        self._opt = None

        if optimizer == 'L-BFGS-B':
            self._use_bfgs = True
        else:
            self._use_bfgs = False
            self._opt = optimizer.minimize(self._loss, var_list=[self._model.get_node('Input').get_data_tensor()])
            self._ses.run(tf.variables_initializer(optimizer.variables()))

    def fit_style_transfer(self, epochs):
        if self._use_bfgs:
            self._opt = get_bfgs(self._loss, self._model.get_node('Input'), epochs)
            self._opt.minimize(self._ses, fetches=[self._loss], loss_callback=update)
        else:
            for i in range(epochs):
                self._ses.run(self._opt)
                ls = self._ses.run(self._loss)
                print(f'Loss in epochs {i+1} is {ls}')

    def get_result(self):
        return self._ses.run(self._model.get_node('Input').get_data_tensor())