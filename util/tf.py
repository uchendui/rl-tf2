import tensorflow as tf


@tf.function
def polyak_average(vars1, vars2, polyak):
    for a, b in zip(vars1, vars2):
        tf.compat.v1.assign(b, polyak * b + (1 - polyak) * a)
