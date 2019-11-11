from __future__ import absolute_import, division, print_function, unicode_literals

#################################################################
# shamelessly lifted from https://github.com/yangyanli/PointCNN
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
DEBUG = False

WITH_GLOBAL = True
KEEP_REMAINDER = True
NUM_CLASS = 40
SAMPLE_NUM = 1024
BATCH_SIZE = 128
#NUM_EPOCHS = 1024
NUM_EPOCHS = 50
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

def get_next(batchDataset):
    return next(iter(batchDataset))

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
    # data_train=data_train[:256,:,:]
    # label_train=label_train[:256]
    data_val, label_val = load_pointcloud(filelist_val)
    # data_val=data_val[:256,:,:]
    # label_val=label_val[:256]
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


def get_no_xforms(xform_num, rotation_range=ROTATION_RANGE, scaling_range=SCALING_RANGE, order=ROTATION_ORDER):
    xforms = np.empty(shape=(xform_num, 3, 3))
    rotations = np.empty(shape=(xform_num, 3, 3))
    for i in range(xform_num):
        rotation = euler2mat(0, 0, 0, order)
        scaling = np.diag([1, 1, 1])

        xforms[i, :] = scaling * rotation 
        rotations[i, :] = rotation
    return xforms, rotations


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
    if(DEBUG == True):
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
        xforms_np, rotations_np = get_xforms(BATCH_SIZE)
        xforms_np = xforms_np.astype('float32')

        if(training == False):
            xforms_np, rotations_np = get_no_xforms(BATCH_SIZE)
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
        return points_augmented

def BatchNormalization(name=''):
    return tf.keras.layers.BatchNormalization( momentum=0.99,
                                        beta_regularizer=tf.keras.regularizers.l2(l=0.5 * (1.0)),
                                        gamma_regularizer=tf.keras.regularizers.l2(l=0.5 * (1.0)),
                                        name=name)

class ProcessingBlock(tf.keras.Model):

    dense_0 = None

    model_a1_0 = None
    model_a1_1 = None
    model_a1_2 = None
    model_a1_3 = None

    model_a2_0 = None
    model_a2_1 = None
    model_a2_2 = None
    model_a2_3 = None

    model_a3_0 = None
    model_a3_1 = None
    model_a3_2 = None
    model_a3_3 = None

    model_b_0 = None
    model_b_1 = None
    model_b_2 = None
    model_b_3 = None

    model_c_0 = None
    model_c_1 = None
    model_c_2 = None
    model_c_3 = None

    model_d_0 = None
    model_d_1 = None
    model_d_2 = None
    model_d_3 = None

    dense_e_0 = None
    dense_e_1 = None
    drop_out_0 = None
    drop_out_1 = None


            
    def define_x_convolution(self, layer_idx, tag, N, K, D, P, C, C_prev, depth_multiplier, with_global):
        def reshape_for_separable_conv2d(input, nN, pP, kK):
            return tf.reshape(input, (nN, pP, kK, kK))
        model_a1 = tf.keras.Sequential([
            tf.keras.Input(shape=(2048, 8, 3,)),
            Dense(output = C // 2, use_bias=False, name=tag + 'nn_fts_from_pts_0_a1_'+str(layer_idx)),
            BatchNormalization(name=tag + 'nn_fts_from_pts_0_bn_a1_'+str(layer_idx)),
            Dense(output = C // 2, use_bias=False,name=tag + 'nn_fts_from_pts_a1_'+str(layer_idx)),
            BatchNormalization(name=tag + 'nn_fts_from_pts_a1_bn_'+str(layer_idx))
        ])

        model_a2 = tf.keras.Sequential([
            tf.keras.Input(shape=(384, 12, 3,)),
            Dense(output = C // 4, use_bias=False, name=tag + 'nn_fts_from_pts_0_a2_'+str(layer_idx)),
            BatchNormalization(name=tag + 'nn_fts_from_pts_0_bn_a2_'+str(layer_idx)),
            Dense(output = C // 4, use_bias=False,name=tag + 'nn_fts_from_pts_a2_'+str(layer_idx)),
            BatchNormalization(name=tag + 'nn_fts_from_pts_bn_a2_'+str(layer_idx))
        ])

        model_a3 = tf.keras.Sequential([
            tf.keras.Input(shape=(128, 16, 3,)),
            Dense(output = C_prev // 4, use_bias=False, name=tag + 'nn_fts_from_pts_0_a3_'+str(layer_idx)),
            BatchNormalization(name=tag + 'nn_fts_from_pts_0_bn_a3_'+str(layer_idx)),
            Dense(output = C_prev // 4, use_bias=False,name=tag + 'nn_fts_from_pts_a3_'+str(layer_idx)),
            BatchNormalization(name=tag + 'nn_fts_from_pts_bn_a3_'+str(layer_idx))
        ])

        if(layer_idx == 0):
            pPP = 2048
            kKK = 8
            PSI = 24
        if(layer_idx == 1):
            pPP = 383
            kKK = 12
            PSI = 60
        if(layer_idx == 2):
            pPP = 128
            kKK = 16
            PSI = 120
        if(layer_idx == 3):
            pPP = 128
            kKK = 6
            PSI = 240

        model_b = tf.keras.Sequential([
                # tf.keras.Input(shape=(pPP, kKK, 3,)),
                Conv2D(K * K, (1,K), name=tag + 'X_0_'+str(layer_idx)),
                BatchNormalization(name=tag + 'X_0_bn_'+str(layer_idx)),
                tf.keras.layers.Lambda(lambda x: reshape_for_separable_conv2d(x,N,P,K)),
                DepthWiseConv2D(K, (1,K), name=tag + 'X_1_'+str(layer_idx)),
                BatchNormalization(name=tag + 'X_1_bn_'+str(layer_idx)),
                tf.keras.layers.Lambda(lambda x: reshape_for_separable_conv2d(x,N,P,K)),
                DepthWiseConv2D(K, (1,K), activation=None,name=tag + 'X_2_'+str(layer_idx)),
                BatchNormalization( name=tag + 'X_2_bn_'+str(layer_idx)),
                tf.keras.layers.Lambda(lambda x: reshape_for_separable_conv2d(x,N,P,K))
        ])

        model_c = tf.keras.Sequential([
            # tf.keras.Input(shape=(pPP, kKK, PSI,)),
            SeparableConv2D(C, (1,K), depth_multiplier=depth_multiplier, name=tag + 'fts_conv_'+str(layer_idx)),
            BatchNormalization( name=tag + 'fts_conv_bn_'+str(layer_idx))
        ])
        model_d = tf.keras.Sequential([
            tf.keras.Input(shape=(pPP, 3,)),
            Dense(C // 4,name=tag + 'fts_global_0_'+str(layer_idx)),
            BatchNormalization(name=tag + 'fts_global_0_bn_'+str(layer_idx)),
            Dense(C // 4,name=tag + 'fts_global_'+str(layer_idx)),
            BatchNormalization(name=tag + 'fts_global_bn_'+str(layer_idx)),
        ])
        return model_a1, model_a2, model_a3, model_b, model_c, model_d         
        
    def __init__(self,input_shape=None, name='processor'):
        super(ProcessingBlock, self).__init__(name=name)
        N=BATCH_SIZE
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
            model_a1, model_a2, model_a3, model_b, model_c, model_d = self.define_x_convolution(layer_idx, tag,  N, K, D, P, C,  C_prev,
                            depth_multiplier, with_global)
            if (layer_idx == 0):
                self.imodel_a1_0 = model_a1
                self.imodel_a2_0 = model_a2
                self.imodel_a3_0 = model_a3
                self.imodel_b_0 = model_b
                self.imodel_c_0 = model_c
                self.imodel_d_0 = model_d
            if (layer_idx == 1):
                self.imodel_a1_1 = model_a1
                self.imodel_a2_1 = model_a2
                self.imodel_a3_1 = model_a3
                self.imodel_b_1 = model_b
                self.imodel_c_1 = model_c
                self.imodel_d_1 = model_d
            if (layer_idx == 2):
                self.imodel_a1_2 = model_a1
                self.imodel_a2_2 = model_a2
                self.imodel_a3_2 = model_a3
                self.imodel_b_2 = model_b
                self.imodel_c_2 = model_c
                self.imodel_d_2 = model_d
            if (layer_idx == 3):
                self.imodel_a1_3 = model_a1
                self.imodel_a2_3 = model_a2
                self.imodel_a3_3 = model_a3
                self.imodel_b_3 = model_b
                self.imodel_c_3 = model_c
                self.imodel_d_3 = model_d
        self.model_a1_0 = self.imodel_a1_0
        self.model_a1_1 = self.imodel_a1_1
        self.model_a1_2 = self.imodel_a1_2
        self.model_a1_3 = self.imodel_a1_3

        self.model_a2_0 = self.imodel_a2_0
        self.model_a2_1 = self.imodel_a2_1
        self.model_a2_2 = self.imodel_a2_2
        self.model_a2_3 = self.imodel_a2_3

        self.model_a3_0 = self.imodel_a3_0
        self.model_a3_1 = self.imodel_a3_1
        self.model_a3_2 = self.imodel_a3_2
        self.model_a3_3 = self.imodel_a3_3

        self.model_b_0 = self.imodel_b_0
        self.model_b_1 = self.imodel_b_1
        self.model_b_2 = self.imodel_b_2
        self.model_b_3 = self.imodel_b_3

        self.model_c_0 = self.imodel_c_0
        self.model_c_1 = self.imodel_c_1
        self.model_c_2 = self.imodel_c_2
        self.model_c_3 = self.imodel_c_3

        self.model_d_0 = self.imodel_d_0
        self.model_d_1 = self.imodel_d_1
        self.model_d_2 = self.imodel_d_2
        self.model_d_3 = self.imodel_d_3


        for fc_idx, fc_param in enumerate(FC_PARAMS):
            C = fc_param['C']
            dropout_rate = fc_param['dropout_rate']
            dense_e = Dense(C, name='fc{:d}'.format(fc_idx))
            drop_out = tf.keras.layers.Dropout(dropout_rate)
            if(fc_idx == 0):
                self.dense_e_0 = dense_e
                self.drop_out_0 = drop_out
            if(fc_idx == 1):
                self.dense_e_1 = dense_e
                self.drop_out_1 = drop_out


    def call_x_convolution(self, layer_idx, pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, atype, is_training, depth_multiplier, with_global=False):
        log('- Running X-Convolution Layer ' + str(layer_idx) + ' : ' + str(N) + ' ' +  str(K) + ' ' + str(P) )
        _, indices_dilated = knn_indices_general(qrs, pts, K * D, True)

        if (layer_idx == 0):
            model_a1 = self.model_a1_0
            model_a2 = self.model_a2_0
            model_a3 = self.model_a3_0
            model_b = self.model_b_0
            model_c = self.model_c_0
            model_d = self.model_d_0
        if (layer_idx == 1):
            model_a1 = self.model_a1_1
            model_a2 = self.model_a2_1
            model_a3 = self.model_a3_1
            model_b = self.model_b_1
            model_c = self.model_c_1
            model_d = self.model_d_1
        if (layer_idx == 2):
            model_a1 = self.model_a1_2
            model_a2 = self.model_a2_2
            model_a3 = self.model_a3_2
            model_b = self.model_b_2
            model_c = self.model_c_2
            model_d = self.model_d_2
        if (layer_idx == 3):
            model_a1 = self.model_a1_3
            model_a2 = self.model_a2_3
            model_a3 = self.model_a3_3
            model_b = self.model_b_3
            model_c = self.model_c_3
            model_d = self.model_d_3

        indices = indices_dilated[:, :, ::D, :]
        nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
        nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
        nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)
        log('- Running model_a: ' )
        if (atype == 1):
            nn_fts_from_pts = model_a1(nn_pts_local, training=is_training)
        elif (atype == 2):
            nn_fts_from_pts = model_a2(nn_pts_local, training=is_training)
        else:
            nn_fts_from_pts = model_a3(nn_pts_local, training=is_training)
        log('- Prepared features to be transformed')
        if fts is None:
            nn_fts_input = nn_fts_from_pts
        else:
            nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
            nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

        log('- Running model_b: ' )
        X_2_KK = model_b(nn_pts_local, training=is_training)
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
        log('- Completed X-transformation')
            
        log('- Running model_c: ' )

        fts_conv = model_c(fts_X, training=is_training)
        fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')
        log('- Completed Separable Convolution 2D')

        if with_global:
            log('- Running model_d: ' )
            fts_global = model_d(qrs, training=is_training)

            log('- Completed concat with global')
            fts_conv_3d = tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')

            return fts_conv_3d
        else:
            return fts_conv_3d

    def call(self, points, training=None):
        log('Processing with training mode '+ str(training))
        is_training = training
        features =  None
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
            if(fc_idx == 0):
                dense_e = self.dense_e_0
                drop_out = self.drop_out_0
            if(fc_idx == 1):
                dense_e = self.dense_e_1
                drop_out = self.drop_out_1
            fc = dense_e(self.fc_layers[-1])
            fc_drop = drop_out(fc, training=is_training)
            self.fc_layers.append(fc_drop)
        return self.fc_layers[-1]
        #, self.logits, self.layer_features

class Summarizer(tf.keras.layers.Layer):
    dense_f = None

    def __init__(self, input_shape=None):
        super(Summarizer, self).__init__()
        dense_f = None
        self.dense_f = Dense(NUM_CLASS, activation=None, name='logits')


    def call(self, fc_layer, training=None):
        log('Summarizing in training mode:' + str(training))
        fc_mean = tf.reduce_mean(fc_layer, axis=1, keepdims=True, name='fc_mean')
        fc_layer = tf.cond(tf.cast(training, tf.bool), lambda: fc_layer, lambda: fc_mean)
        output = self.dense_f(fc_mean)
        # output = tf.nn.softmax(output, name='probs')
        return output


class PointCNN(tf.keras.Sequential):

    layer_fts = None
    fc_layers = None

    def __init__(self, point_num, input_shape):
        super(PointCNN, self).__init__()
        self.point_num = point_num
        # self.augment = Augment( point_num, input_shape=[2048,6])
        # self.process = ProcessingBlock(input_shape=input_shape)
        # self.summarize = Summarizer()
        self.add(Augment( point_num, input_shape=[2048,6]))
        self.add(ProcessingBlock(input_shape=[2048, 3]))
        self.add(Summarizer())

    # def build(self, input_shape):
        # super(PointCNN, self).build(input_shape)
        # self.augment.build(input_shape)
        # self.process.build(input_shape)
        # self.summarize.build()

    def call(self, points, training=None):
        log('Running model with training mode' +  str(training))
        x =  super(PointCNN, self).call(points, training=training)
        return x

        # x = self.augment(points)
        # self.fc_layers, _ = self.process(x, training=training)
        # return x

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
    train_batch_num_per_epoch = num_train // BATCH_SIZE
    train_batch_num = train_batch_num_per_epoch * NUM_EPOCHS
    log('{:d} training batches'.format(train_batch_num))

    test_batch_num_per_epoch = num_test // BATCH_SIZE
    test_batch_num = test_batch_num_per_epoch * NUM_EPOCHS
    log('{:d} testing batches'.format(test_batch_num))

    train_dataset = tf.data.Dataset.from_tensor_slices((data_train, label_train))
    buffer_size = num_train * NUM_EPOCHS 
    train_ds = train_dataset.repeat(NUM_EPOCHS).shuffle(BATCH_SIZE * 4).batch(BATCH_SIZE, drop_remainder=True)
    val_buffer_size = num_test * NUM_EPOCHS
    test_dataset = tf.data.Dataset.from_tensor_slices((data_val, label_val))
    test_ds = test_dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE, drop_remainder=True)


    model = PointCNN(point_num,(2048,6))
    model.summary()
    # model.get_layer('processing_block').summary()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE_BASE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True)

    class CustomLoss(tf.keras.losses.Loss): 
        def __init__(self):
            super(CustomLoss, self).__init__()

        def call(self, y_true, y_pred):
            actuals = tf.cast(tf.reshape(y_true,[-1]),tf.int64)
            onehot = tf.one_hot(actuals, NUM_CLASS, axis=-1)
            probs = tf.nn.softmax(y_pred, name='probs')
            probs = tf.reshape(probs, [BATCH_SIZE, NUM_CLASS])
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False).call(onehot, probs)
            return loss

    class CustomAccuracy(tf.keras.metrics.Accuracy): 
        def __init__(self, name='pccnAccuracy', **kwargs):
            super(CustomAccuracy, self).__init__(name=name, **kwargs)
            self.accuracy = self.add_weight(name='acc', initializer='zeros',dtype='float32')

        def update_state(self, y_true, y_pred, sample_weight=None):
            actuals = tf.cast(tf.reshape(y_true,[-1]),tf.int64)
            probs = tf.nn.softmax(y_pred, name='probs')
            predictions = tf.argmax(probs, axis=-1, name='predictions')
            super(CustomAccuracy, self).update_state(actuals,predictions)


        def result(self):
            return super(CustomAccuracy, self).result()

        def reset_states(self):
            super(CustomAccuracy, self).reset_states()            
            
    # tf.maximum(lr_schedule, LEARNING_RATE_MIN)
    adam = tf.keras.optimizers.Adam(
                    epsilon=EPSILON,
                    learning_rate=lr_schedule)
    model.compile(optimizer='adam',
                loss=CustomLoss(),
                # loss='sparse_categorical_crossentropy',
                metrics=[CustomAccuracy()])
    log('Compiled model')
    model.build((128,2048,3))
    log('Built model')

    ########################################
    history = model.fit_generator(
        train_ds,
        steps_per_epoch=train_batch_num_per_epoch,
        epochs=NUM_EPOCHS,
        verbose=1,
        validation_data=test_ds,
        validation_steps=test_batch_num_per_epoch
    )
    log('Completed running  model')

    acc = history.history['pccnAccuracy']
    val_acc = history.history['val_pccnAccuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    log('Accuracy ' + ' ' + str(acc) + ' ' + str(val_acc))
    sys.exit()
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
    step = 0
    loss = 0
    t_1_acc = 0
    t_1_per_class_acc = 0

    test_step = 0
    test_loss = 0
    test_t_1_acc = 0
    test_t_1_per_class_acc = 0

    def reset_metrics():
        loss = 0
        t_1_acc = 0
        t_1_per_class_acc = 0
        log('resetting metrics')

    def reset_test_metrics():
        test_loss = 0
        test_t_1_acc = 0
        test_t_1_per_class_acc = 0
        log('resetting test_metrics')

    def train():
        log('training')
    
    def test():
        log('testing')

    def update_metrics():
        log('updating metrics')
    
    def update_test_metrics():
        log('updating test metrics')

    def aggregate_metrics():
        log('aggregating metrics')

    def aggregate_test_metrics():
        log('aggregating test metrics')

    for epoch_num in range(NUM_EPOCHS):
        for batch_num in range(BATCH_SIZE):
            # Training
            offset = int(random.gauss(0, SAMPLE_NUM * SAMPLE_NUM_VARIANCE))
            offset = max(offset, -SAMPLE_NUM * SAMPLE_NUM_CLIP)
            offset = min(offset, SAMPLE_NUM * SAMPLE_NUM_CLIP)
            sample_num_train = SAMPLE_NUM + offset
            xforms_np, rotations_np = get_xforms(BATCH_SIZE)
            reset_metrics()
            train()
            update_metrics()
            if batch_idx_train % 10 == 0:
                aggregate_metrics()
                step += 1
                log(' [Train]-Iter: {:06d}  Loss: {:.4f}  T-1 Acc: {:.4f}  T-1 mAcc: {:.4f}'
                            .format(step, loss, t_1_acc, t_1_per_class_acc))

            # Testing
            if ((batch_num % STEP_VAL == 0 and batch_num != 0 ) or (batch_num == batch_num - 1)):
                next_test_ds = get_next(test_ds)
                reset_metrics()
                test()

    
main()






