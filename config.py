#-*- coding: utf-8 -*-
import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network


net_arg = add_argument_group('Network')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default=['lsp_dataset', 'mpii_human_pose_v1', 'Human3.6M', 'mpi_inf_3dhp', 'data_256'])  # Human3.6M authorization faied...
data_arg.add_argument('--batch_size', type=int, default=1)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--j', type=int, default=15)
train_arg.add_argument('--lr', type=float, default=2.5e-4)
train_arg.add_argument('--lr_mtplr', type=float, default=1.e-3)
train_arg.add_argument('--beta1', type=float, default=0.0)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--two_d_pose', type=int, default=100)
train_arg.add_argument('--three_d_pose', type=int, default=100)
train_arg.add_argument('--minibatch', type=int, default=4)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=1000)
misc_arg.add_argument('--save_step', type=int, default=2000)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='C:/Users/tigerkim/Documents/GitHub/data')
misc_arg.add_argument('--check_dir', type=str, default='check')
misc_arg.add_argument('--pre_dir', type=str, default='pre_trained')


def get_config():
    config, unparsed = parser.parse_known_args()
    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    setattr(config, 'data_format', data_format)
    return config, unparsed
