import unittest
import hypothesis as hy
import hypothesis.strategies as st
# import hypothesis.extra.numpy as hynp
# import numpy as np
# from keras import backend as K
# from keras.models import Model
# from keras.layers import Activation, Convolution2D, Dropout, GlobalAveragePooling2D, Concatenate, Dense, Input, AveragePooling2D
# from keras.layers.normalization import BatchNormalization
# from keras.regularizers import l2
# from pytescian import types as an
from densenet import DenseNet, dense_block, convolution_block, transition_layer, __version__


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(input_shape=st.one_of(st.tuples(st.integers(min_value=30,
        max_value=70), st.integers(min_value=30, max_value=70), st.integers
        (min_value=1, max_value=3)), st.just(None)), dense_blocks=st.one_of
        (st.integers(min_value=1, max_value=5), st.just(3)), dense_layers=
        st.one_of(st.integers(min_value=-1, max_value=5), st.just(-1)),
        growth_rate=st.one_of(st.integers(min_value=1, max_value=20), st.
        just(12)), nb_classes=st.one_of(st.integers(min_value=2, max_value=
        22), st.just(None)), dropout_rate=st.one_of(st.floats(min_value=0,
        max_value=1, allow_nan=None, allow_infinity=None, width=64,
        exclude_min=True, exclude_max=True), st.just(None)), bottleneck=st.
        one_of(st.sampled_from([True, False]), st.just(False)), compression
        =st.one_of(st.floats(min_value=0, max_value=1, allow_nan=None,
        allow_infinity=None, width=64, exclude_min=True, exclude_max=False),
        st.just(1.0)), weight_decay=st.one_of(st.floats(min_value=0.0001,
        max_value=0.01, allow_nan=None, allow_infinity=None, width=64,
        exclude_min=False, exclude_max=False), st.just(0.0001)), depth=st.
        one_of(st.integers(min_value=10, max_value=100), st.just(40)))
    @hy.settings(deadline=None, suppress_health_check=[hy.HealthCheck.
        filter_too_much, hy.HealthCheck.too_slow])
    def test_DenseNet(self, input_shape, dense_blocks, dense_layers,
        growth_rate, nb_classes, dropout_rate, bottleneck, compression,
        weight_decay, depth):
        hy.assume(not nb_classes == None)
        hy.assume(not (compression <= 0.0 or compression > 1.0))
        hy.assume(not (type(dense_layers) is list and len(dense_layers) !=
            dense_blocks))
        hy.assume(not input_shape is None)
        DenseNet(input_shape, dense_blocks, dense_layers, growth_rate,
            nb_classes, dropout_rate, bottleneck, compression, weight_decay,
            depth)
