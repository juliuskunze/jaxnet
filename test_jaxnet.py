from jax import numpy as np, random, jit

from jaxnet import Dense, Sequential, relu, parameterized, Conv, flatten, MaxPool, zeros


def test_params():
    net = Dense(2, kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 3))

    params = net.init_params(inputs, random.PRNGKey(0))
    assert len(params) == 2
    assert np.array_equal(params.kernel, np.zeros((3, 2)))
    assert np.array_equal(params.bias, np.zeros(2))

    output = net(params, inputs)
    assert np.array_equal(output, np.zeros((1, 2)))

    output = jit(net)(params, inputs)
    assert np.array_equal(output, np.zeros((1, 2)))


def test_submodule():
    @parameterized
    def net(inputs, layer=Dense(2, zeros, zeros)):
        return layer(inputs)

    inputs = np.zeros((1, 2))

    params = net.init_params(inputs, random.PRNGKey(0))
    assert len(params) == 1
    assert len(params.layer) == 2
    assert np.array_equal(params.layer.kernel, np.zeros((2, 2)))
    assert np.array_equal(params.layer.bias, np.zeros(2))

    output = net(params, inputs)
    assert np.array_equal(output, np.zeros((1, 2)))

    output = jit(net)(params, inputs)
    assert np.array_equal(output, np.zeros((1, 2)))


def test_submodule_list():
    layer = Sequential([Dense(2, zeros, zeros), relu])
    inputs = np.zeros((1, 2))

    params = layer.init_params(inputs, random.PRNGKey(0))
    assert len(params) == 1
    assert len(params.layers) == 2
    assert np.array_equal(params.layers[0].kernel, np.zeros((2, 2)))
    assert np.array_equal(params.layers[0].bias, np.zeros(2))
    assert params.layers[1] == ()

    output = layer(params, inputs)
    assert np.array_equal(output, np.zeros((1, 2)))

    output = jit(layer)(params, inputs)
    assert np.array_equal(output, np.zeros((1, 2)))


def test_internal_param_sharing():
    @parameterized
    def shared_net(inputs, layer=Dense(2, zeros, zeros)):
        return layer(layer(inputs))

    inputs = np.zeros((1, 2))
    params = shared_net.init_params(inputs, random.PRNGKey(0))
    assert len(params) == 1
    assert len(params.layer) == 2
    assert np.array_equal(params.layer.kernel, np.zeros((2, 2)))
    assert np.array_equal(params.layer.bias, np.zeros(2))

    output = shared_net(params, inputs)
    assert np.array_equal(output, np.zeros((1, 2)))

    output = jit(shared_net)(params, inputs)
    assert np.array_equal(output, np.zeros((1, 2)))


def test_external_param_sharing():
    layer = Dense(2, zeros, zeros)
    shared_net = Sequential([layer, layer])

    inputs = np.zeros((1, 2))
    params = shared_net.init_params(inputs, random.PRNGKey(0))
    assert len(params) == 1
    assert len(params.layers) == 2
    assert np.array_equal(params.layers[0].kernel, np.zeros((2, 2)))
    assert np.array_equal(params.layers[0].bias, np.zeros(2))
    assert params.layers[1] == ()

    output = shared_net(params, inputs)
    assert np.array_equal(output, np.zeros((1, 2)))

    output = jit(shared_net)(params, inputs)
    assert np.array_equal(output, np.zeros((1, 2)))


def test_internal_param_sharing2():
    @parameterized
    def shared_net(inputs, layer=Sequential([Dense(2, zeros, zeros), relu])):
        inputs = layer(inputs)
        return layer(inputs)

    inputs = np.zeros((1, 2))
    params = shared_net.init_params(inputs, random.PRNGKey(0))

    assert len(params) == 1
    assert len(params.layer) == 1
    assert len(params.layer.layers) == 2
    assert len(params.layer.layers[0]) == 2
    assert np.array_equal(params.layer.layers[0].kernel, np.zeros((2, 2)))
    assert np.array_equal(params.layer.layers[0].bias, np.zeros(2))

    output = shared_net(params, inputs)
    assert np.array_equal(output, np.zeros((1, 2)))


def test_conv_flatten():
    conv = Conv(2, filter_shape=(3, 3), padding='SAME', kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 5, 5, 2))

    params = conv.init_params(inputs, random.PRNGKey(0))
    output = conv(params, inputs)
    assert np.array_equal(output, np.zeros((1, 5, 5, 2)))

    flattened = Sequential([conv, flatten])
    output = flattened({'layers': [params, ()]}, inputs)
    assert np.array_equal(output, np.zeros((1, 50)))


def test_conv_max_pool():
    conv = Conv(2, filter_shape=(3, 3), padding='SAME', kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 5, 5, 2))

    params = conv.init_params(inputs, random.PRNGKey(0))
    pooled = Sequential([conv, MaxPool(window_shape=(1, 1), strides=(2, 2))])
    inputs = np.zeros((1, 5, 5, 2))
    output = pooled({'layers': [params, ()]}, inputs)
    assert np.array_equal(output, np.zeros((1, 3, 3, 2)))


def test_example():
    net = Sequential([Dense(2), relu, Dense(4)])
    batch = np.zeros((3, 2))
    params = net.init_params(batch, random.PRNGKey(0))
    print(params.layers[2].bias)

    output = net(params, batch)
    output = jit(net)(params, batch)
