import matplotlib.pyplot as plt
import np as np
import numpy as np
import keras
import cv2
from keras.layers import Input
from keras.models import Model
import random
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf
import os.path
import pandas as pd
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, AveragePooling2D, Dense, Dropout, Flatten
from sklearn.metrics import roc_curve, auc


TESTING_IMAGES_FOLDER = "testing/"
WEIGHTS_FOLDER = "weights/"

n_classes = 14

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
img_input = Input(shape=input_shape)

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth,num_classes):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu', name='last_activation_layer')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='sigmoid',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def print_roc_curve(gen, model, filename):
    pred_labels = model.predict(gen, verbose = 1)
    fig, ax = plt.subplots(1, 1, figsize = (9, 9))
    auc_list = []
    for (idx, class_label) in n_classes:
        fpr, tpr, thresholds = roc_curve(gen.labels[:, idx].astype(int), pred_labels[:, idx])
        cur_auc = auc(fpr, tpr)
        auc_list.append(cur_auc)
        ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (class_label, cur_auc))
    ax.legend()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    fig.savefig(filename)
    auc_list = np.array(auc_list)
    return auc_list


print("** load model **")
multi_disease_model = resnet_v2((128, 128, 1), (9*6)+2, n_classes)
multi_disease_model.compile(optimizer='adam')
multi_disease_model.summary()


print("** load weights **")
multi_disease_model.load_weights(os.path.join(WEIGHTS_FOLDER, "weights_epoch_20.h5"))
print(multi_disease_model.get_weights()[0])
print("\n\n")

auc_epochs = []




Xtest = []
# Data Read
for test_img_name in os.listdir(TESTING_IMAGES_FOLDER):
    img = cv2.imread(os.path.join(TESTING_IMAGES_FOLDER, test_img_name))
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.
    img = np.mean(img, axis=2)  # convert to 1-dim gray
    print("**load image**")
    Xtest.append(img)
    # Xtest = Xtest.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # demonstrate prediction
    # print("file: ", test_img_name)

print("Epoch 5")
multi_disease_model.load_weights(os.path.join(WEIGHTS_FOLDER, "weights_epoch_20.h5"))
auc_5 = print_roc_curve(Xtest, multi_disease_model, 'val_set_roc_curve_epoch_20.png')
auc_epochs.append(auc_5)


'''yhat = multi_disease_model.predict(Xtest, verbose=1)
print("predictions: ")
print(np.round(yhat, 2))'''