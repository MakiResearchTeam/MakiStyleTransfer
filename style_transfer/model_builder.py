from makiflow.layers import *
from makiflow.models import Classificator
from .utils import InputLayer_mod


def create_model_vgg16(input_shape, xinp=None, mode='Network', number_of_layers=15):
    assert number_of_layers > 2 or number_of_layers < 19
    if mode == 'Network':
        in_x = InputLayer(input_shape=input_shape, name='Input')
    elif mode == 'Style':
        in_x = InputLayer_mod(input_shape=input_shape, name='Input', value=xinp)
    else:
        raise TypeError(f'{mode} can not be used')

    layers = []
    spec_output = []

    layers.append(ConvLayer(kw=3, kh=3, in_f=3, out_f=64, name='conv1/conv1_1')(in_x))
    layers.append(ConvLayer(kw=3, kh=3, in_f=64, out_f=64, name='conv1/conv1_2'))
    layers.append(AvgPoolLayer(name='block1_pool', padding='VALID'))
    layers.append(ConvLayer(kw=3, kh=3, in_f=64, out_f=128, name='conv2/conv2_1'))
    layers.append(ConvLayer(kw=3, kh=3, in_f=128, out_f=128, name='conv2/conv2_2'))
    layers.append( AvgPoolLayer(name='block2_pool', padding='VALID'))
    layers.append( ConvLayer(kw=3, kh=3, in_f=128, out_f=256, name='conv3/conv3_1'))
    layers.append(ConvLayer(kw=3, kh=3, in_f=256, out_f=256, name='conv3/conv3_2'))
    layers.append(ConvLayer(kw=3, kh=3, in_f=256, out_f=256, name='conv3/conv3_3'))
    layers.append(AvgPoolLayer(name='block3_pool', padding='VALID'))
    layers.append(ConvLayer(kw=3, kh=3, in_f=256, out_f=512, name='conv4/conv4_1'))
    layers.append(ConvLayer(kw=3, kh=3, in_f=512, out_f=512, name='conv4/conv4_2'))
    layers.append(ConvLayer(kw=3, kh=3, in_f=512, out_f=512, name='conv4/conv4_3'))
    layers.append(AvgPoolLayer(name='block4_pool', padding='VALID'))
    layers.append(ConvLayer(kw=3, kh=3, in_f=512, out_f=512, name='conv5/conv5_1'))
    layers.append(ConvLayer(kw=3,kh=3,in_f=512,out_f=512,name='conv5/conv5_2'))
    layers.append(ConvLayer(kw=3,kh=3,in_f=512,out_f=512,name='conv5/conv5_3'))
    layers.append(AvgPoolLayer(name='block5_pool', padding='VALID'))

    x = layers[0]
    spec_output.append(x)

    for i in range(1, number_of_layers):
        x = layers[i](x)

        if x.get_name()[-1] == '1':
            spec_output.append(x)
    if x.get_name() != spec_output[-1].get_name():
        spec_output.append(x)

    # names for loading weights
    names = [i for i, _ in x.get_previous_tensors().items()] + [x.get_name()]

    return Classificator(in_x, x), in_x, x, names, spec_output

