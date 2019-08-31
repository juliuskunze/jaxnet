def grad(fun):
    pass # TODO


def square(x):
    return x * x


_ = grad(square) # make equivalent to:
def square_grad(x):
    return 2 * x


def multiply(a, b):
    return a * b


_ grad(multiply) # make equivalent to:
def multiply_grad(a, b):
    return b, a
