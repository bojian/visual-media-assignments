"""
For visual media assignment(2018/08/08) 
Implement resnet-34 via keras

The code is modified from:
    https://blog.csdn.net/googler_offer/article/details/79521453 (2018/08/08 accessed)

The function of identity shortcuts is modified:
1. When the dimensions increase, 1×1 convolutions is used for matching dimensions
2. In a building block, the second relu function is performed after the addition of two outputs.

Reference:
     K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
     (https://arxiv.org/abs/1512.03385)
"""


# coding=utf-8
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, Activation, ZeroPadding2D
from keras.layers import add, Flatten
import numpy as np

seed = 7
np.random.seed(seed)

"""
#the original code
def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
 
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x

"""


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    """
    #the original code

    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x
    """
    x = Conv2D(nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(inpt)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(nb_filter=nb_filter, kernel_size=kernel_size, strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    if with_conv_shortcut:
        # 1×1 convolutions is used for matching dimensions
        shortcut = Conv2D(nb_filter=nb_filter, strides=strides, kernel_size=(1,1))(inpt)
        shortcut = BatchNormalization(axis=3)(shortcut)
        # First add, then relu
        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x
    else:
        # First add, then relu
        x = add([x, inpt])
        x = Activation('relu')(x)
        return x


inpt0 = Input(shape=(224, 224, 3))
x = ZeroPadding2D((3, 3))(inpt0)
x = Conv2D(nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(x)
#The original code
#x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')
x = BatchNormalization(axis=3)(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
# (56,56,64)
x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
# (28,28,128)
x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
# (14,14,256)
x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
# (7,7,512)
x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)
x = Dense(1000, activation='softmax')(x)

model = Model(inputs=inpt0, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()