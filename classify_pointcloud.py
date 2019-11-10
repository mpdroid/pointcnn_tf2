from __future__ import absolute_import, division, print_function, unicode_literals

#################################################################
# shamelessly lifted from https://github.com/yangyanli/PointCNN
#    I bow to thee, O gods of deep learning
#################################################################


import os
import sys
import math
import random
import shutil
import argparse
import importlib
import data_utils
import numpy as np
from datetime import datetime
import tensorflow as tf
from transforms3d.euler import euler2mat
import h5py
import plyfile
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WITH_GLOBAL = True
KEEP_REMAINDER = True
NUM_CLASS = 40
SAMPLE_NUM = 1024
BATCH_SIZE = 128
NUM_EPOCHS = 1024
STEP_VAL = 500
LEARNING_RATE_BASE = 0.01
DECAY_STEPS = 8000
DECAY_RATE = 0.5
LEARNING_RATE_MIN = 1e-6
WEIGHT_DECAY = 1e-5

JITTER = 0.0
JITTER_VAL = 0.0

ROTATION_RANGE = [0, math.pi, 0, 'u']
ROTATION_RANGE_VAL = [0, 0, 0, 'u']
ROTATION_ORDER = 'rxyz'

SCALING_RANGE = [0.1, 0.1, 0.1, 'g']
SCALING_RANGE_VAL = [0, 0, 0, 'u']

SAMPLE_NUM_VARIANCE = 1 // 8
SAMPLE_NUM_CLIP = 1 // 4

x = 3
XCONV_PARAM_NAMES = ('K', 'D', 'P', 'C', 'links')
XCONV_PARAMS = [dict(zip(XCONV_PARAM_NAMES, xconv_param)) for xconv_param in
                [(8, 1, -1, 16 * x, []),
                 (12, 2, 384, 32 * x, []),
                 (16, 2, 128, 64 * x, []),
                 (16, 3, 128, 128 * x, [])]]

WITH_GLOBAL = True

FC_PARAM_NAMES = ('C', 'dropout_rate')
FC_PARAMS = [dict(zip(FC_PARAM_NAMES, fc_param)) for fc_param in
             [(128 * x, 0.0),
              (64 * x, 0.2)]]

SAMPLING = 'random'

OPTIMIZER = 'adam'
EPSILON = 1e-2

DATA_DIM = 6
USE_EXTRA_FEATURES = False
SORTING_METHOD = None

POOL_SETTING_TRAIN = None
POOL_SETTING_VAL = None


def grouped_shuffle(inputs):
    for idx in range(len(inputs) - 1):
        assert (len(inputs[idx]) == len(inputs[idx + 1]))

    shuffle_indices = np.arange(inputs[0].shape[0])
    np.random.shuffle(shuffle_indices)
    outputs = []
    for idx in range(len(inputs)):
        outputs.append(inputs[idx][shuffle_indices, ...])
    return outputs

def load_pointcloud(filelist):
    points = []
    labels = []
    count=5

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        filename = os.path.basename(line.rstrip())
        data = h5py.File(os.path.join(folder, filename))
        if 'normal' in data:
            points.append(np.concatenate([data['data'][...], data['normal'][...]], axis=-1).astype(np.float32))
        else:
            points.append(data['data'][...].astype(np.float32))
        labels.append(np.squeeze(data['label'][:]).astype(np.int64))
    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0))

def load_datasets(filelist, filelist_val):
    data_train, label_train = grouped_shuffle(load_pointcloud(filelist))
    # data_train=data_train[:640,:,:]
    # label_train=label_train[:640]
    data_val, label_val = load_pointcloud(filelist_val)
    # data_val=data_val[:640,:,:]
    # label_val=label_val[:640]
    return data_train, label_train, data_val, label_val


def gauss_clip(mu, sigma, clip):
    v = random.gauss(mu, sigma)
    v = max(min(v, mu + clip * sigma), mu - clip * sigma)
    return v


def uniform(bound):
    return bound * (2 * random.random() - 1)

def scaling_factor(scaling_param, method):
    try:
        scaling_list = list(scaling_param)
        return random.choice(scaling_list)
    except:
        if method == 'g':
            return gauss_clip(1.0, scaling_param, 3)
        elif method == 'u':
            return 1.0 + uniform(scaling_param)


def rotation_angle(rotation_param, method):
    try:
        rotation_list = list(rotation_param)
        return random.choice(rotation_list)
    except:
        if method == 'g':
            return gauss_clip(0.0, rotation_param, 3)
        elif method == 'u':
            return uniform(rotation_param)



#def get_xforms(xform_num, rotation_range=(0, 0, 0, 'u'), scaling_range=(0.0, 0.0, 0.0, 'u'), order='rxyz'):
def get_xforms(xform_num, rotation_range=ROTATION_RANGE, scaling_range=SCALING_RANGE, order=ROTATION_ORDER):
    xforms = np.empty(shape=(xform_num, 3, 3))
    rotations = np.empty(shape=(xform_num, 3, 3))
    for i in range(xform_num):
        rx = rotation_angle(rotation_range[0], rotation_range[3])
        ry = rotation_angle(rotation_range[1], rotation_range[3])
        rz = rotation_angle(rotation_range[2], rotation_range[3])
        rotation = euler2mat(rx, ry, rz, order)

        sx = scaling_factor(scaling_range[0], scaling_range[3])
        sy = scaling_factor(scaling_range[1], scaling_range[3])
        sz = scaling_factor(scaling_range[2], scaling_range[3])
        scaling = np.diag([sx, sy, sz])

        xforms[i, :] = scaling * rotation
        rotations[i, :] = rotation
    return xforms, rotations


def log(message):
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print(time_string + ' : ', message )

def Dense(output, use_bias=False, activation=tf.nn.elu, name=''):
    return tf.keras.layers.Dense(units=output, activation=activation,
                            kernel_initializer='glorot_normal',
                            kernel_regularizer=tf.keras.regularizers.l2(1.0),
                            use_bias=use_bias, name=name)

def Conv2D(filters, kernel_size, activation=tf.nn.elu, use_bias=False, name=''):
    return tf.keras.layers.Conv2D( filters, kernel_size=kernel_size, strides=(1, 1), padding='VALID',
                              activation=activation,
                              kernel_initializer='glorot_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(1.0),
                              use_bias=use_bias, name=name)

def DepthWiseConv2D(depth_multiplier, kernel_size, use_bias=False, activation=tf.nn.elu, name=''):
    return tf.keras.layers.SeparableConv2D(filters=depth_multiplier*depth_multiplier,
				kernel_size=kernel_size,
				strides=(1,1),
				padding='valid',
				data_format=None,
				dilation_rate=(1,1),
				depth_multiplier=depth_multiplier,
				activation=activation,
				use_bias=use_bias,
				depthwise_initializer='glorot_normal',
				pointwise_initializer='glorot_normal',
				bias_initializer='zeros' if use_bias else None,
				depthwise_regularizer=tf.keras.regularizers.l2(1.0),
				pointwise_regularizer=tf.keras.regularizers.l2(1.0),
				bias_regularizer=tf.keras.regularizers.l2(1.0)  if use_bias else None,
                activity_regularizer=None,
				depthwise_constraint=None,
				pointwise_constraint=None,
                bias_constraint=None,
                name=name
			)

def BatchNormalization(training=True, name=''):
    return tf.keras.layers.BatchNormalization( momentum=0.99,
                                         beta_regularizer=tf.keras.regularizers.l2(l=0.5 * (1.0)),
                                         gamma_regularizer=tf.keras.regularizers.l2(l=0.5 * (1.0)),
                                         name=name)

def SeparableConv2D(output,  kernel_size, use_bias=False, depth_multiplier=1,
                      activation=tf.nn.elu, name=''):
    return tf.keras.layers.SeparableConv2D(output, kernel_size=kernel_size, strides=(1, 1), padding='VALID',
                                        activation=activation,
                                        depth_multiplier=depth_multiplier,
                                        depthwise_initializer='glorot_normal',
                                        pointwise_initializer='glorot_normal',
                                        depthwise_regularizer=tf.keras.regularizers.l2(l=1.0),
                                        pointwise_regularizer=tf.keras.regularizers.l2(l=1.0),
                                        use_bias=use_bias,
                                        name=name)

def layer_log(input, message):
    log('after layer: '+ message)
    log(input.shape)
    log(input.dtype)
    log(input.shape[-1])
    return input


# A shape is (N, C)
def distance_matrix(A):
    r = tf.reduce_sum(input_tensor=A * A, axis=1, keepdims=True)
    m = tf.matmul(A, tf.transpose(a=A))
    D = r - 2 * m + tf.transpose(a=r)
    return D


# A shape is (N, P, C)
def batch_distance_matrix(A):
    r = tf.reduce_sum(input_tensor=A * A, axis=2, keepdims=True)
    m = tf.matmul(A, tf.transpose(a=A, perm=(0, 2, 1)))
    D = r - 2 * m + tf.transpose(a=r, perm=(0, 2, 1))
    return D


# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_distance_matrix_general(A, B):
    r_A = tf.reduce_sum(input_tensor=A * A, axis=2, keepdims=True)
    r_B = tf.reduce_sum(input_tensor=B * B, axis=2, keepdims=True)
    m = tf.matmul(A, tf.transpose(a=B, perm=(0, 2, 1)))
    D = r_A - 2 * m + tf.transpose(a=r_B, perm=(0, 2, 1))
    return D

def tf_unique(one_d_tensor):
   return tf.unique(one_d_tensor[0], out_idx=tf.dtypes.int64)

# A shape is (N, P, C)
def find_duplicate_columns(A):
    N = A.shape[0]
    P = A.shape[1]
    # indices_duplicated = tf.Variable(tf.ones(shape=(N, 1, P), dtype=tf.dtypes.int64))
    indices_duplicated = tf.ones(shape=(N, 1, P), dtype=tf.dtypes.int64)
    for idx in range(N):
        _, indices = tf_unique(A[idx])
        ab_1 = tf.constant([idx, 0], tf.int64)
        l = indices.shape[0]
        ab = tf.broadcast_to(ab_1, [l,2])
        c = tf.reshape(indices, [l,1])
        abc = tf.concat([ab, c],1)
        values = tf.zeros(shape=abc.shape[0], dtype=tf.int64)
        delta = tf.SparseTensor(abc,values,indices_duplicated.shape)
        indices_duplicated += tf.sparse.to_dense(delta)
    return indices_duplicated

# add a big value to duplicate columns
def prepare_for_unique_top_k(D, A):
    indices_duplicated = find_duplicate_columns(A)
    D += tf.reduce_max(input_tensor=D)*tf.cast(indices_duplicated, tf.float32)


# return shape is (N, P, K, 2)
def knn_indices(points, k, sort=True, unique=True):
    points_shape = tf.shape(input=points)
    batch_size = points_shape[0]
    point_num = points_shape[1]

    D = batch_distance_matrix(points)
    if unique:
        prepare_for_unique_top_k(D, points)
    distances, point_indices = tf.nn.top_k(-D, k=k, sorted=sort)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
    return -distances, indices

def knn_indices_general(queries, points, k, sort=True, unique=True):
    queries_shape = tf.shape(input=queries)
    batch_size = queries_shape[0]
    point_num = queries_shape[1]

    D = batch_distance_matrix_general(queries, points)
    if unique:
        prepare_for_unique_top_k(D, points)
    distances, point_indices = tf.nn.top_k(-D, k=k, sorted=sort)  # (N, P, K)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
    return -distances, indices


def x_convolution(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, depth_multiplier, with_global=False):
    log('Defining X-Convolution: ' + str(N) + ' ' +  str(K) + ' ' + str(P) )
    _, indices_dilated = knn_indices_general(qrs, pts, K * D, True)
    indices = indices_dilated[:, :, ::D, :]
  
    def reshape_for_separable_conv2d(input, N, P, K):
        return tf.reshape(input, (N, P, K, K))

    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)
    log('Defining model_a: ' )

    # Prepare features to be transformed
    model_a = tf.keras.Sequential([
        Dense(output = C_pts_fts, use_bias=False, name=tag + 'nn_fts_from_pts_0'),
        BatchNormalization(training=is_training,name=tag + 'nn_fts_from_pts_0_bn'),
        Dense(output = C_pts_fts, use_bias=False,name=tag + 'nn_fts_from_pts_bn'),
        BatchNormalization(training=is_training,name=tag + 'nn_fts_from_pts_bn')
    ])
    log('Defined model_a: ' )
    nn_fts_from_pts = model_a(nn_pts_local, training=is_training)
    log('Prepared features to be transformed')
    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    model_b = tf.keras.Sequential([
            Conv2D(K * K, (1,K), name=tag + 'X_0'),
            BatchNormalization(training=is_training, name=tag + 'X_0_bn'),
            tf.keras.layers.Lambda(lambda x: reshape_for_separable_conv2d(x,N,P,K)),
            DepthWiseConv2D(K, (1,K), name=tag + 'X_1'),
            BatchNormalization(training=is_training, name=tag + 'X_1_bn'),
            tf.keras.layers.Lambda(lambda x: reshape_for_separable_conv2d(x,N,P,K)),
            DepthWiseConv2D(K, (1,K), activation=None,name=tag + 'X_2'),
            BatchNormalization(training=is_training, name=tag + 'X_2_bn'),
            tf.keras.layers.Lambda(lambda x: reshape_for_separable_conv2d(x,N,P,K))
    ])
    X_2_KK = model_b(nn_pts_local, training=is_training)
    fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
    log('Completed X-transformation')

    model_c = tf.keras.Sequential([
        SeparableConv2D(C, (1,K), depth_multiplier=depth_multiplier, name=tag + 'fts_conv'),
        BatchNormalization(training=is_training, name=tag + 'X_2_bn')
    ])
    fts_conv = model_c(fts_X, training=is_training)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')
    log('Completed Separable Convolution 2D')

    if with_global:
        model_d = tf.keras.Sequential([
            Dense(C // 4,name=tag + 'fts_global_0'),
            BatchNormalization(training=is_training,name=tag + 'fts_global_0_bn'),
            Dense(C // 4,name=tag + 'fts_global'),
            BatchNormalization(training=is_training,name=tag + 'fts_global_bn'),
        ])
        fts_global = model_d(qrs, training=is_training)

        log('Completed concat with global')
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d



# the returned indices will be used by tf.gather_nd
def get_indices( batch_size, sample_num, point_num ):
    point_nums = tf.fill([batch_size], point_num)
    # if not isinstance(point_num, np.ndarray):
    #     point_nums = np.full((batch_size), point_num)
    # else:
    #     point_nums = point_num

    indices = []
    for i in range(batch_size):
        pt_num = point_nums[i]
        pool_size = pt_num
        if pool_size > sample_num:
            choices = np.random.choice(pool_size, sample_num, replace=False)
        else:
            choices = np.concatenate((np.random.choice(pool_size, pool_size, replace=False),
                                      np.random.choice(pool_size, sample_num - pool_size, replace=True)))
        if pool_size < pt_num:
            choices_pool = np.random.choice(pt_num, pool_size, replace=False)
            choices = choices_pool[choices]
        choices = np.expand_dims(choices, axis=1)
        choices_2d = np.concatenate((np.full_like(choices, i), choices), axis=1)
        indices.append(choices_2d)

    return np.stack(indices)
    
    
def augment(points, xforms, range=None):
    points_xformed = tf.matmul(points, xforms, name='points_xformed')
    if range is None:
        return points_xformed

    jitter_data = range * tf.random.normal(tf.shape(input=points_xformed), name='jitter_data')
    jitter_clipped = tf.clip_by_value(jitter_data, -5 * range, 5 * range, name='jitter_clipped')
    return points_xformed + jitter_clipped



class Augment(tf.keras.layers.Layer):
    def __init__(self, point_num, input_shape=None):
        super(Augment, self).__init__(input_shape=input_shape)
        self.point_num = point_num


    def call(self, inputs, training=None):
        log('Augmenting with training mode:' + str(training))
        N = tf.shape(input=inputs)[0]
        xforms_np, rotations_np = get_xforms(BATCH_SIZE)
        xforms_np = xforms_np.astype('float32')
        # indices = get_indices(BATCH_SIZE, sample_num_train, self.point_num)
        # points_features_sampled = tf.gather_nd(inputs, indices=indices, name='features_sampled')
        points_features_sampled = inputs
        if DATA_DIM > 3:
            points_sampled, features_sampled = tf.split(points_features_sampled,
                                                        [3, DATA_DIM - 3],
                                                        axis=-1,
                                                        name='split_points_features')
        else:
            points_sampled = inputs
        points_augmented = augment(points_sampled, xforms_np, np.array([JITTER_VAL]))
        # points_augmented = points_sampled
        return points_augmented

class ProcessingLayer(tf.keras.layers.Layer):

    dense_0 = None
    model_a1s = [None] * len(XCONV_PARAMS)
    model_a2s = [None] * len(XCONV_PARAMS)
    model_a3s = [None] * len(XCONV_PARAMS)
    model_bs = [None] * len(XCONV_PARAMS)
    model_cs = [None] * len(XCONV_PARAMS)
    model_ds = [None] * len(XCONV_PARAMS)
    dense_es = [None] * len(FC_PARAMS)
    drop_outs = [None] * len(FC_PARAMS)
    dense_f = None

    def __init__(self, features):
        super(ProcessingLayer, self).__init__()
        self.features = features
    
    def define_x_convolution(self, tag, N, K, D, P, C, C_prev, depth_multiplier, with_global):
        def reshape_for_separable_conv2d(input, N, P, K):
            return tf.reshape(input, (N, P, K, K))
        model_a1 = tf.keras.Sequential([
            Dense(output = C // 2, use_bias=False, name=tag + 'nn_fts_from_pts_0'),
            BatchNormalization(name=tag + 'nn_fts_from_pts_0_bn'),
            Dense(output = C // 2, use_bias=False,name=tag + 'nn_fts_from_pts_bn'),
            BatchNormalization(name=tag + 'nn_fts_from_pts_bn')
        ])

        model_a2 = tf.keras.Sequential([
            Dense(output = C // 4, use_bias=False, name=tag + 'nn_fts_from_pts_0'),
            BatchNormalization(name=tag + 'nn_fts_from_pts_0_bn'),
            Dense(output = C // 4, use_bias=False,name=tag + 'nn_fts_from_pts_bn'),
            BatchNormalization(name=tag + 'nn_fts_from_pts_bn')
        ])

        model_a3 = tf.keras.Sequential([
            Dense(output = C_prev // 4, use_bias=False, name=tag + 'nn_fts_from_pts_0'),
            BatchNormalization(name=tag + 'nn_fts_from_pts_0_bn'),
            Dense(output = C_prev // 4, use_bias=False,name=tag + 'nn_fts_from_pts_bn'),
            BatchNormalization(name=tag + 'nn_fts_from_pts_bn')
        ])

        model_b = tf.keras.Sequential([
                Conv2D(K * K, (1,K), name=tag + 'X_0'),
                BatchNormalization(name=tag + 'X_0_bn'),
                tf.keras.layers.Lambda(lambda x: reshape_for_separable_conv2d(x,N,P,K)),
                DepthWiseConv2D(K, (1,K), name=tag + 'X_1'),
                BatchNormalization(name=tag + 'X_1_bn'),
                tf.keras.layers.Lambda(lambda x: reshape_for_separable_conv2d(x,N,P,K)),
                DepthWiseConv2D(K, (1,K), activation=None,name=tag + 'X_2'),
                BatchNormalization( name=tag + 'X_2_bn'),
                tf.keras.layers.Lambda(lambda x: reshape_for_separable_conv2d(x,N,P,K))
        ])
        model_c = tf.keras.Sequential([
            SeparableConv2D(C, (1,K), depth_multiplier=depth_multiplier, name=tag + 'fts_conv'),
            BatchNormalization( name=tag + 'X_2_bn')
        ])
        model_d = tf.keras.Sequential([
            Dense(C // 4,name=tag + 'fts_global_0'),
            BatchNormalization(name=tag + 'fts_global_0_bn'),
            Dense(C // 4,name=tag + 'fts_global'),
            BatchNormalization(name=tag + 'fts_global_bn'),
        ])
        return model_a1, model_a2, model_a3, model_b, model_c, model_d               


    def build(self, input_shape):
        N = input_shape[0]
        C_fts = XCONV_PARAMS[0]['C'] // 2
        self.dense_0 = Dense(C_fts, name='features_hd')
        for layer_idx, layer_param in enumerate(XCONV_PARAMS):
            tag = 'xconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K']
            D = layer_param['D']
            P = layer_param['P']
            C = layer_param['C']
            C_prev = XCONV_PARAMS[layer_idx - 1]['C']
            links = layer_param['links']
            if layer_idx == 0:
                depth_multiplier = 4
            else:   
                depth_multiplier = math.ceil(C / C_prev)
            with_global = (WITH_GLOBAL and layer_idx == len(XCONV_PARAMS) - 1)
            model_a1, model_a2, model_a3, model_b, model_c, model_d = self.define_x_convolution(tag, N, K, D, P, C,  C_prev,
                            depth_multiplier, with_global)
            self.model_a1s[layer_idx] = model_a1
            self.model_a2s[layer_idx] = model_a2
            self.model_a3s[layer_idx] = model_a3
            self.model_bs[layer_idx] = model_b
            self.model_cs[layer_idx] = model_c
            self.model_ds[layer_idx] = model_d
        for fc_idx, fc_param in enumerate(FC_PARAMS):
            C = fc_param['C']
            dropout_rate = fc_param['dropout_rate']
            dense_e = Dense(C, name='fc{:d}'.format(fc_idx))
            self.dense_es[fc_idx] = dense_e
            drop_out = tf.keras.layers.Dropout(dropout_rate)
            self.drop_outs[fc_idx] = drop_out

        self.dense_f = Dense(NUM_CLASS, activation=None, name='logits')
            

    def call_x_convolution(self, layer_idx, pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, atype, is_training, depth_multiplier, with_global=False):
        log('- Running X-Convolution Layer ' + str(layer_idx) + ' : ' + str(N) + ' ' +  str(K) + ' ' + str(P) )
        _, indices_dilated = knn_indices_general(qrs, pts, K * D, True)
        indices = indices_dilated[:, :, ::D, :]
        nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
        nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
        nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)
        log('- Running model_a: ' )
        if (atype == 1):
            nn_fts_from_pts = self.model_a1s[layer_idx](nn_pts_local, training=is_training)
        elif (atype == 2):
            nn_fts_from_pts = self.model_a2s[layer_idx](nn_pts_local, training=is_training)
        else:
            nn_fts_from_pts = self.model_a3s[layer_idx](nn_pts_local, training=is_training)
        log('- Prepared features to be transformed')
        if fts is None:
            nn_fts_input = nn_fts_from_pts
        else:
            nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
            nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

        X_2_KK = self.model_bs[layer_idx](nn_pts_local, training=is_training)
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
        log('- Completed X-transformation')


        fts_conv = self.model_cs[layer_idx](fts_X, training=is_training)
        fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')
        log('- Completed Separable Convolution 2D')

        if with_global:
            fts_global = self.model_ds[layer_idx](qrs, training=is_training)

            log('Completed concat with global')
            return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
        else:
            return fts_conv_3d

    def call(self, points, features, training=None):
        log('Processing with training mode '+ str(training))
        is_training = training
        features = self.features
        N = tf.shape(input=points)[0]
        self.layer_points = [points]
        if features is None:
            self.layer_features = [features]
        else:
            features = tf.reshape(features, (N, -1, DATA_DIM - 3), name='features_reshape')
            C_fts = XCONV_PARAMS[0]['C'] // 2
            features_hd = self.dense_0(features)
            self.layer_features = [features_hd]
        for layer_idx, layer_param in enumerate(XCONV_PARAMS):
            tag = 'xconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K']
            D = layer_param['D']
            P = layer_param['P']
            C = layer_param['C']
            links = layer_param['links']
            pts = self.layer_points[-1]
            fts = self.layer_features[-1]
            if P == -1 or (layer_idx > 0 and P == XCONV_PARAMS[layer_idx - 1]['P']):
                qrs = self.layer_points[-1]
            else: 
                qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
            self.layer_points.append(qrs)
            atype = 1
            if layer_idx == 0:
                atype = 1 if fts is None else 2
                C_pts_fts = C // 2 if fts is None else C // 4
                depth_multiplier = 4
            else:
                atype = 3
                C_prev = XCONV_PARAMS[layer_idx - 1]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)

            with_global = (WITH_GLOBAL and layer_idx == len(XCONV_PARAMS) - 1)
            fts_xconv = self.call_x_convolution(layer_idx, pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, atype,
                            is_training, depth_multiplier, with_global)

            fts_list = []

            for link in links:
                fts_from_link = self.layer_features[link]
                if fts_from_link is not None:
                    fts_slice = tf.slice(fts_from_link, (0, 0, 0), (-1, P, -1), name=tag + 'fts_slice_' + str(-link))
                    fts_list.append(fts_slice)
            if fts_list:
                fts_list.append(fts_xconv)
                self.layer_features.append(tf.concat(fts_list, axis=-1, name=tag + 'fts_list_concat'))
            else:
                self.layer_features.append(fts_xconv)

        self.fc_layers = [self.layer_features[-1]]
        for fc_idx, fc_param in enumerate(FC_PARAMS):
            fc = self.dense_es[fc_idx](self.fc_layers[-1])
            fc_drop = self.drop_outs[fc_idx](fc, training=True)
            self.fc_layers.append(fc_drop)
        return self.fc_layers, self.layer_features
        #, self.logits, self.layer_features



class PointCNN(tf.keras.Model):

    layer_fts = None
    fc_layers = None
    dense_f = None

    def __init__(self, point_num):
        super(PointCNN, self).__init__()
        self.point_num = point_num
        self.augment = Augment( point_num, input_shape=(2048, 6))
        self.process = ProcessingLayer(self.layer_fts)

    def build(self, input_shape):
        self.augment.build(input_shape)
        self.process.build(input_shape)
        self.dense_f = Dense(NUM_CLASS, activation=None, name='logits')
    def call(self, points, training=None):
        log('Running model with training mode' +  str(training))
        x = self.augment(points)
        self.fc_layers, self.layer_fts = self.process(x, self.layer_fts, training=training)
        fc_layer = self.fc_layers[-1]
        fc_mean = tf.reduce_mean(fc_layer, axis=1, keepdims=True, name='fc_mean')
        self.fc_layers[-1] = tf.cond(tf.cast(training, tf.bool), lambda: fc_layer, lambda: fc_mean)
        x = self.dense_f(self.fc_layers[-1])
        #x = self.dense_f(fc_mean)
        # tf.print('output shape:',tf.shape(x))
        # tf.print('logits:',x[0][0])
        # tf.print('logits:',tf.reduce_sum(x[0][0])
        return x

    # def __call__(self, points):
    #     return self.call(points,training=training)

######


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', help='Path to data', required=True)
    parser.add_argument('--path_val', '-v', help='Path to validation data')
    args = parser.parse_args()
    data_train, label_train, data_val, label_val = load_datasets(args.path, args.path_val)

    num_train = data_train.shape[0]
    point_num = data_train.shape[1]
    num_test = data_val.shape[0]

    log('{:d}/{:d} training/validation samples.'.format(num_train, num_test))
    train_batch_num_per_epoch = math.ceil(num_train / BATCH_SIZE)
    train_batch_num = train_batch_num_per_epoch * NUM_EPOCHS
    log('{:d} training batches'.format(train_batch_num))

    test_batch_num_per_epoch = math.ceil(num_test / BATCH_SIZE)
    test_batch_num = test_batch_num_per_epoch * NUM_EPOCHS
    log('{:d} testing batches'.format(test_batch_num))

    train_dataset = tf.data.Dataset.from_tensor_slices((data_train, label_train))
    train_ds = train_dataset.shuffle(BATCH_SIZE * 4).batch(BATCH_SIZE, drop_remainder=True)
    #.batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((data_val, label_val)).batch(BATCH_SIZE, drop_remainder=True)
    #.batch(BATCH_SIZE)

    refmodel = tf.keras.models.Sequential()
    refmodel.add(tf.keras.layers.Conv2D(2048, (6, 6), activation='relu', input_shape=(None, 2048, 6)))
    refmodel.add(tf.keras.layers.MaxPooling2D((5, 5)))
    refmodel.add(tf.keras.layers.Conv2D(4096, (6, 6), activation='relu'))
    refmodel.add(tf.keras.layers.MaxPooling2D((2, 2)))
    refmodel.add(tf.keras.layers.Conv2D(4096, (6, 6), activation='relu'))

    model = PointCNN(point_num)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE_BASE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True)

    class CustomLoss(tf.keras.losses.CategoricalCrossentropy): 
        def __init__(self):
            super(CustomLoss, self).__init__(from_logits=False)

    # def custom_loss(y_true, y_pred):
    #     reg_loss = WEIGHT_DECAY * tf.compat.v1.losses.get_regularization_loss()
    #     return tf.losses.CategoricalCrossentropy(labels=labels_tile, logits=model.logits) 


    # tf.maximum(lr_schedule, LEARNING_RATE_MIN)
    adam = tf.keras.optimizers.Adam(
                    epsilon=EPSILON,
                    learning_rate=lr_schedule)
    model.compile(optimizer='adam',
                loss=CustomLoss(),
                # loss='sparse_categorical_crossentropy',
                metrics=['accuracy','categorical_accuracy'])
    log('Compiled model')
    model.build([128,2048,3])
    history = model.fit_generator(
        train_ds,
        steps_per_epoch=train_batch_num_per_epoch,
        epochs=NUM_EPOCHS,
        verbose=1,
        validation_data=test_ds,
        validation_steps=test_batch_num_per_epoch
    )
    log('Completed running  model')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    log('Accuracy ' + ' ' + str(acc) + ' ' + str(val_acc))
    epochs_range = range(NUM_EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()

main()






