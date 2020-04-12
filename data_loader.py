import tensorflow as tf
import glob
import numpy as np
import scipy.io as sio
import skvideo.io
import cv2
'''
lsp -> image and its joint-painted image.  -> image and its joint(15)
mpii_human_pose_v1 -> image and its attribute.  -> image and its joint(15)
Human3.6M -> authorization fail. so ignore
mpi_inf_3dhp -> video/forgground/annote -> video -> select only one -> hadamard forground and add places2 -> joint(15)
'''


def get_loader(path, dataset, batch_size):
    with tf.compat.v1.variable_scope("tfData"):
        lsp_image = glob.glob("{}/{}/images/*.{}".format(path, dataset[0], "jpg"))  # lsp_dataset/images\
        res = sio.loadmat('{}/{}/joints'.format(path, dataset[0]))  # 3X14X2000
        lsp_jointd = np.transpose(res['joints'], (2, 0, 1))
        whole_queue_0 = tf.data.Dataset.from_tensor_slices((lsp_image, lsp_jointd))
        whole_queue = tf.data.Dataset.zip(whole_queue_0)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        whole_queue = whole_queue.shuffle(buffer_size=1001)
        whole_queue = whole_queue.repeat()
        whole_queue = whole_queue.map(preprocess_lsp, num_parallel_calls=AUTOTUNE)
        whole_queue = whole_queue.batch(batch_size)
        whole_queue = whole_queue.prefetch(buffer_size=AUTOTUNE)
    return whole_queue


def get_3d_loader(path, dataset, batch_size):
    '''
    # print(res['joints'])
    # mpii_image = glob.glob("{}/{}/mpii_human_pose_v1.tar/images/*.{}".format(path, dataset[0], "jpg"))  # mpii_human_pose_v1.tar\images
    res = sio.loadmat(
        "{}/{}".format(path, 'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1'))
    # print(res['RELEASE']['annolist'][0][0][0][0][0][0][0][0])  # name
    # print(res['RELEASE']['annolist'][0][0][1])
    '''
    ChairMasks = glob.glob("{}/*".format('D:/mpi_inf_3dhp/S1/Seq1/ChairMasks_npy'))
    FGmasks = glob.glob("{}/*".format('D:/mpi_inf_3dhp/S1/Seq1/FGmasks_npy'))
    imageSequence = glob.glob("{}/*".format('D:/mpi_inf_3dhp/S1/Seq1/imageSequence_npy'))
    res = sio.loadmat('{}/annot'.format('D:/mpi_inf_3dhp/S1/Seq1'))
    for i in range(8):
        if i is 0:
            mpi = res['annot3'][0][0]
        else:
            mpi = np.concatenate((mpi, res['annot3'][i][0]), axis=0)
    # print(chair_mask.shape, fg_mask.shape, image_sequence.shape, mpi.shape)
    whole_queue_0 = tf.data.Dataset.from_tensor_slices((ChairMasks, FGmasks, imageSequence, mpi))
    whole_queue = tf.data.Dataset.zip(whole_queue_0)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    whole_queue = whole_queue.shuffle(buffer_size=9)
    whole_queue = whole_queue.repeat()
    whole_queue = whole_queue.map(preprocess_mpi, num_parallel_calls=AUTOTUNE)
    whole_queue = whole_queue.batch(batch_size)
    whole_queue = whole_queue.prefetch(buffer_size=AUTOTUNE)
    return whole_queue


def preprocess_mpi(ChairMasks, FGmasks, imageSequence, mpi):
    c_mask = tf.numpy_function(read_npy_file, [ChairMasks], tf.float32)
    f_mask = tf.numpy_function(read_npy_file, [FGmasks], tf.float32)
    video = tf.numpy_function(read_npy_file, [imageSequence], tf.float32)
    print(c_mask.shape, f_mask.shape, video.shape, mpi.shape)
    return ChairMasks, FGmasks, imageSequence, mpi


def read_video_file(video):
    file = cv2.VideoCapture(video)
    return file.astype(np.float32)


def preprocess_lsp(image, joint):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)  # [HWC] joint[3 ,14], but only need [2, 14]
    joint_image = joint_to_image(image, joint)  # [2, 14]  -> [HW(14)]
    y = tf.reduce_min(joint[1, :])  # min_value = y
    x = tf.reduce_min(joint[0, :])  # min_value = y
    h = tf.reduce_max(joint[1, :]) - y  # should be 20%
    w = tf.reduce_max(joint[1, :]) - x  # should be 40%
    y = tf.cast(tf.round(tf.maximum(y - 0.2 * h, 1.)), dtype=tf.int32)
    x = tf.cast(tf.round(tf.maximum(x - 0.4 * w, 1.)), dtype=tf.int32)
    h = tf.minimum(tf.cast(tf.round(1.2 * h), dtype=tf.int32), tf.shape(image)[1] - y - 1)
    w = tf.minimum(tf.cast(tf.round(1.4 * w), dtype=tf.int32), tf.shape(image)[0] - x - 1)
    image = tf.image.crop_to_bounding_box(image, y, x, h, w)
    joint_image = tf.image.crop_to_bounding_box(joint_image, y, x, h, w)
    image = tf.image.resize(image, [256, 256])
    joint_image = tf.image.resize(joint_image, [256, 256])
    return image, joint_image


def read_npy_file(video):
    file = np.load(video)
    return file.astype(np.float32)


def check_file_shape(x):
    print('================')
    print('total_image_length :' + str(len(x[0])))
    print('total_joint_length :' + str(len(x[1])))
    print(x[0][0], x[1][0].shape)  # [dynamicl_ength], [3, 14]
    # y, sr = librosa.load(x[0][0], sr=100)
    # print('audio shape: ' + str(y.shape[0]) + ", sample rate: " + str(sr))  # 298, 100 not this # 65664, 22050
    # file = np.load(x[0][0])
    # print('video shape: ' + str(file.shape))  # 298
    # file = np.load(x[1][0])
    # print('video shape: ' + str(file.shape))  # 288, 360, 3, 75  # 64, 80, 3, 75  # 64, 80, 1, 75
    print('================')


def joint_to_image(image, joint):
    # input: [2, 14]
    joint = tf.cast(tf.math.round(joint), dtype=tf.float32)
    joint_image_y = tf.expand_dims(tf.cast(tf.range(tf.shape(image)[0]), dtype=tf.float32), -1)
    joint_image_y = tf.maximum(1. - tf.abs(joint_image_y - tf.expand_dims(joint[1], 0)), 0.)

    joint_image_x = tf.expand_dims(tf.cast(tf.range(tf.shape(image)[1]), dtype=tf.float32), -1)
    joint_image_x = tf.maximum(1. - tf.abs(joint_image_x - tf.expand_dims(joint[0], 0)), 0.)
    joint_image_x = tf.transpose(tf.expand_dims(joint_image_x, 0), [2, 0, 1])
    joint_image_y = tf.transpose(tf.expand_dims(joint_image_y, 1), [2, 0, 1])
    joint_image = tf.transpose(tf.matmul(joint_image_y, joint_image_x), [1, 2, 0])
    return joint_image
