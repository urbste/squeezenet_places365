import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import FlattenLayer
from lasagne.layers import rrelu
from lasagne.nonlinearities import rectify, softmax, leaky_rectify
from lasagne.layers import ConcatLayer
from lasagne.init import Glorot, GlorotUniform


#def fire_expand_block(input_layer, name, nr_filter_squeeze, nr_filter_expand):
#    
#    net = {}
#    # squeeze
#    net[name+'squeeze1x1'] = ConvLayer(input_layer, 
#                                            nr_filter_squeeze, 1, pad='same',nonlinearity=None, W=GlorotUniform('relu'))
#    net[name+'relu_squeeze1x1'] = BatchNormLayer(NonlinearityLayer(net[name+'squeeze1x1'],nonlinearity=rectify))
#    
#    # expand left
#    net[name+'expand1x1'] = ConvLayer(net[name+'relu_squeeze1x1'], 
#                                            nr_filter_expand, 1, pad='same', nonlinearity=None, W=GlorotUniform('relu'))
#    net[name+'relu_expand1x1'] = BatchNormLayer(NonlinearityLayer(net[name+'expand1x1'],nonlinearity=rectify))
#    
#    # expand right    
#    net[name+'expand3x3'] = ConvLayer(net[name+'relu_squeeze1x1'], 
#                                            nr_filter_expand, 3, pad='same', nonlinearity=None,  W=GlorotUniform('relu'))
#    net[name+'relu_expand3x3'] = BatchNormLayer(NonlinearityLayer(net[name+'expand3x3'],nonlinearity=rectify))
#
#    return net
    
def fire_expand_block(input_layer, name, nr_filter_squeeze, nr_filter_expand):
    
    net = {}
    # squeeze
    net[name+'squeeze1x1'] = BatchNormLayer(ConvLayer(input_layer, 
                                            nr_filter_squeeze, 1, pad='same',nonlinearity=None, W=GlorotUniform('relu')))
    net[name+'relu_squeeze1x1'] = rrelu(net[name+'squeeze1x1'])
    
    # expand left
    net[name+'expand1x1'] = BatchNormLayer(ConvLayer(net[name+'relu_squeeze1x1'], 
                                            nr_filter_expand, 1, pad='same', nonlinearity=None, W=GlorotUniform('relu')))
    net[name+'relu_expand1x1'] = rrelu(net[name+'expand1x1'])
    
    # expand right    
    net[name+'expand3x3'] = BatchNormLayer(ConvLayer(net[name+'relu_squeeze1x1'], 
                                            nr_filter_expand, 3, pad='same', nonlinearity=None,  W=GlorotUniform('relu')))
    net[name+'relu_expand3x3'] = rrelu(net[name+'expand3x3'])

    return net
    
def build_squeeznetv2(inputTensor, block_names, output_dim=1000):
    net = {}
     
    net['input'] = InputLayer((None, 3, 224, 224),inputTensor)
    net['conv1'] = BatchNormLayer(ConvLayer(net['input'], 64, 3, stride=(2,2), pad='same', nonlinearity=None, W=GlorotUniform('relu')))
    net['relu_conv1'] = rrelu(net['conv1'])
    net['pool1'] = PoolLayer(net['relu_conv1'], 3, stride=(2,2), mode='max')
    
    sub_net = fire_expand_block(net['pool1'], block_names[0], nr_filter_squeeze=16, nr_filter_expand=64)
    net.update(sub_net)
    #fire2/concat
    net[block_names[0]+'concat'] = ConcatLayer([sub_net[block_names[0]+'relu_expand1x1'],sub_net[block_names[0]+'relu_expand3x3']], axis = 1)
     
    sub_net= fire_expand_block(net[block_names[0]+'concat'], block_names[1], nr_filter_squeeze=16, nr_filter_expand=64)
    net.update(sub_net)
    #fire3/concat
    net[block_names[1]+'concat'] = ConcatLayer([sub_net[block_names[1]+'relu_expand1x1'],sub_net[block_names[1]+'relu_expand3x3']], axis = 1)
    
    net['pool3'] = PoolLayer(net[block_names[1]+'concat'], 3, stride=(2,2), mode='max')
    
    sub_net = fire_expand_block(net['pool3'], block_names[2], nr_filter_squeeze=32, nr_filter_expand=128)
    net.update(sub_net)
    #fire4/concat
    net[block_names[2]+'concat'] = ConcatLayer([sub_net[block_names[2]+'relu_expand1x1'],sub_net[block_names[2]+'relu_expand3x3']], axis = 1) 
    
    sub_net = fire_expand_block(net[block_names[2]+'concat'], block_names[3], nr_filter_squeeze=32, nr_filter_expand=128)
    net.update(sub_net)
    #fire5/concat
    net[block_names[3]+'concat'] = ConcatLayer([sub_net[block_names[3]+'relu_expand1x1'],sub_net[block_names[3]+'relu_expand3x3']], axis = 1)
    
    net['pool5'] = PoolLayer(net[block_names[3]+'concat'], 3, stride=(2,2), mode='max')
    
    sub_net = fire_expand_block(net['pool5'], block_names[4], nr_filter_squeeze=48, nr_filter_expand=192)
    net.update(sub_net)
    #fire6/concat
    net[block_names[4]+'concat'] = ConcatLayer([sub_net[block_names[4]+'relu_expand1x1'],sub_net[block_names[4]+'relu_expand3x3']], axis = 1)
     
    sub_net = fire_expand_block(net[block_names[4]+'concat'], block_names[5], nr_filter_squeeze=48, nr_filter_expand=192)
    net.update(sub_net)
    #fire7/concat
    net[block_names[5]+'concat'] = ConcatLayer([sub_net[block_names[5]+'relu_expand1x1'],sub_net[block_names[5]+'relu_expand3x3']], axis = 1)
    
    sub_net = fire_expand_block(net[block_names[5]+'concat'], block_names[6], nr_filter_squeeze=64, nr_filter_expand=256)
    net.update(sub_net)
    #fire8/concat
    net[block_names[6]+'concat'] = ConcatLayer([sub_net[block_names[6]+'relu_expand1x1'],sub_net[block_names[6]+'relu_expand3x3']], axis = 1)
    
    sub_net = fire_expand_block(net[block_names[6]+'concat'], block_names[7], nr_filter_squeeze=64, nr_filter_expand=256)
    net.update(sub_net)
    #fire9/concat
    net[block_names[7]+'concat'] = ConcatLayer([sub_net[block_names[7]+'relu_expand1x1'],sub_net[block_names[7]+'relu_expand3x3']], axis = 1)
    
    net['drop9'] = DropoutLayer(net[block_names[7]+'concat'], p = 0.5)
    
    net['conv10'] = ConvLayer(net['drop9'], output_dim, 1, pad='same', nonlinearity=None, W=GlorotUniform('relu'))
    net['relu_conv10'] = rrelu(net['conv10'])
    
    net['pool10'] = PoolLayer(net['relu_conv10'], 13, pad=0, mode='average_exc_pad', ignore_border=False, stride=1)
    
    net['prob'] = NonlinearityLayer(FlattenLayer(net['pool10']), nonlinearity=softmax)
    
    return net