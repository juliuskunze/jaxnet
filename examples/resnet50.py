# Run this example in your browser: https://colab.research.google.com/drive/1q6yoK_Zscv-57ZzPM4qNy3LgjeFzJ5xN#scrollTo=p0J1g94IpxK-

import numpy.random as npr
from jax import jit, grad, random, numpy as np
from jax.experimental import optimizers

from jaxnet import Conv, BatchNorm, GeneralConv, MaxPool, Dense, AvgPool, flatten, logsoftmax, \
    Sequential, relu, parametrized, InputDependent


def ConvBlock(kernel_size, filters, strides=(2, 2)):
    ks = kernel_size
    filters1, filters2, filters3 = filters

    @parametrized
    def conv_block(inputs,
                   main=Sequential(
                       Conv(filters1, (1, 1), strides), BatchNorm(), relu,
                       Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), relu,
                       Conv(filters3, (1, 1)), BatchNorm()),
                   shortcut=Sequential(Conv(filters3, (1, 1), strides), BatchNorm())):
        return relu(sum((main(inputs), shortcut(inputs))))

    return conv_block


def IdentityBlock(kernel_size, filters):
    ks = kernel_size
    filters1, filters2 = filters

    def make_main(inputs):
        # the number of output channels depends on the number of input channels
        return Sequential(
            Conv(filters1, (1, 1)), BatchNorm(), relu,
            Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), relu,
            Conv(inputs.shape[3], (1, 1)), BatchNorm())

    @parametrized
    def identity_block(inputs, main=InputDependent(make_main)):
        return relu(sum((main(inputs), inputs)))

    return identity_block


def ResNet50(num_classes):
    return Sequential(
        GeneralConv(('HWCN', 'OIHW', 'NHWC'), 64, (7, 7), (2, 2), 'SAME'),
        BatchNorm(), relu, MaxPool((3, 3), strides=(2, 2)),
        ConvBlock(3, [64, 64, 256], strides=(1, 1)),
        IdentityBlock(3, [64, 64]),
        IdentityBlock(3, [64, 64]),
        ConvBlock(3, [128, 128, 512]),
        IdentityBlock(3, [128, 128]),
        IdentityBlock(3, [128, 128]),
        IdentityBlock(3, [128, 128]),
        ConvBlock(3, [256, 256, 1024]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        ConvBlock(3, [512, 512, 2048]),
        IdentityBlock(3, [512, 512]),
        IdentityBlock(3, [512, 512]),
        AvgPool((7, 7)), flatten, Dense(num_classes), logsoftmax)


if __name__ == "__main__":
    rng_key = random.PRNGKey(0)

    batch_size = 8
    num_classes = 1001
    input_shape = (224, 224, 3, batch_size)
    step_size = 0.1
    num_steps = 10

    resnet = ResNet50(num_classes)


    @parametrized
    def loss(inputs, targets, resnet=resnet):
        logits = resnet(inputs)
        return np.sum(logits * targets)


    @parametrized
    def accuracy(inputs, targets, resnet=resnet):
        target_class = np.argmax(targets, axis=-1)
        predicted_class = np.argmax(resnet(inputs), axis=-1)
        return np.mean(predicted_class == target_class)


    def synth_batches():
        rng = npr.RandomState(0)
        while True:
            images = rng.rand(*input_shape).astype('float32')
            labels = rng.randint(num_classes, size=(batch_size, 1))
            onehot_labels = labels == np.arange(num_classes)
            yield images, onehot_labels


    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)
    batches = synth_batches()


    @jit
    def update(i, opt_state, inputs, targets):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, inputs, targets), opt_state)


    opt_state = opt_init(loss.init_params(rng_key, *next(batches)))
    for i in range(num_steps):
        opt_state = update(i, opt_state, *next(batches))
    trained_params = get_params(opt_state)