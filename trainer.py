from __future__ import print_function

import os

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from tqdm import trange
from models import *
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.mobilenet import mobilenet_v2


class Trainer(object):
    def __init__(self, config, get_loader):

        self.config = config
        self.get_loader = get_loader
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.j = config.j
        self.lr = tf.Variable(config.lr, name='lr')
        self.lr_mtplr = tf.Variable(config.lr_mtplr, name='lr_mtplr')

        self.model_dir = config.model_dir
        self.load_path = config.load_path
        self.summary_path = os.path.join(config.ckpt_dir)
        self.pre_dir = config.data_dir[0] + '/' + config.pre_dir

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        self.start_step = 0
        self.log_step = config.log_step
        self.save_step = config.save_step
        self.k_t = tf.Variable(0., trainable=False)

        data = self.get_loader
        self.iterator = data.make_initializable_iterator()
        self.data = self.iterator.get_next()

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.compat.v1.train.Saver()
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.summary_path)

        sv = tf.compat.v1.train.Supervisor(logdir=self.model_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_model_secs=1200,
                                 global_step=self.step,
                                 ready_for_local_init_op=None)

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if not self.is_train:
            g = tf.get_default_graph()
            g._finalized = False

    def train(self):
        self.initial_iterator()
        self.init_fn(self.sess)
        # teacher first, student last
        # 2d train first, 3d last
        for step in trange(self.start_step, self.max_step):
            if step < self.two_d_pose:
                fetch_dict = {
                    '2d_t': self.two_d_teacher_optim,
                }
            elif step < self.two_d_pose + self.three_d_pose:
                if step is self.two_d_pose:
                    self.data_change()
                fetch_dict = {
                    '3d_t': self.three_d_teachter_optim,
                }
            elif step < self.two_d_pose + self.three_d_pose + self.two_d_pose:
                fetch_dict = {
                    '2d_s': self.two_d_teacher_optim,
                }
            else:
                fetch_dict = {
                    '3d_s': self.three_d_student_optim,
                }
            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "teacher_heat_map_loss": self.t_hm_loss,
                    "teacher_local_map_loss": self.t_lm_loss,
                    "student_heat_map_loss": self.s_hm_loss,
                    "student_local_map_loss": self.s_lm_loss,
                })
            result = self.sess.run(fetch_dict)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()
                t_hm_loss = result['teacher_heat_map_loss']
                t_lm_loss = result['teacher_local_map_loss']
                s_hm_loss = result['student_heat_map_loss']
                s_lm_loss = result['student_local_map_loss']
                print(
                    "[{}/{}] Loss_FD: [{:.6f}] Loss_SD: [{:.6f}] || Loss_G: [{:.6f}]" \
                        .format(step, self.max_step, t_hm_loss, t_lm_loss, s_hm_loss, s_lm_loss))

    def build_model(self):
        with tf.compat.v1.variable_scope('preprocessing'):
            two_d_x, two_d_gt  = self.preprocessing()
            x, gt = two_d_x, two_d_gt

        batch_input = utils.upscale_to(x, [224, 224], 'NCHW')
        batch_input, batch_mean_rgb = self.mean_rgb(batch_input)
        t, self.t_var = self.teacher(batch_input, self.j)
        s, self.s_var = self.student(batch_input, self.j)
        # print(t, s)
        with tf.compat.v1.variable_scope("loss_design"):
            with tf.compat.v1.variable_scope("optimizer"):
                two_d_optimizer = tf.compat.v1.train.RMSPropOptimizer(self.lr)
                three_d_optimizer = tf.compat.v1.train.AdamOptimizer(self.lr * self.lr_mtplr)

            with tf.compat.v1.variable_scope("loss_function"):
                t = utils.upscale_to_bil(t, [256, 256], 'NCHW')
                s = utils.upscale_to_bil(s, [256, 256], 'NCHW')
                t_h, t_l = tf.split(t, [self.j, -1], axis=1)  # heat for joint, x/y/z for joint
                s_h, s_l = tf.split(s, [self.j, -1], axis=1)  # heat for joint, x/y/z for joint
                gt_h, gt_l = two_d_gt, two_d_gt  # but data has only x/y... where is heat and z
                print(t_h, t_l, s_h, s_l, gt_h, gt_l)
                with tf.compat.v1.variable_scope("t_loss"):
                    self.t_hm_loss = tf.reduce_mean(tf.pow(tf.abs(t_h - gt_h), 2.))
                    self.t_lm_loss = tf.reduce_mean(tf.pow(tf.abs(t_l - gt_l), 2.))
                with tf.compat.v1.variable_scope("s_loss"):
                    alpha = 0.5
                    self.s_hm_loss = tf.reduce_mean(alpha * tf.pow(tf.abs(s_h - gt_h), 2.)
                                            + (1.-alpha) * tf.pow(tf.abs(s_h - t_h), 2.)
                    )
                    self.s_lm_loss = tf.reduce_mean(alpha * tf.pow(gt_h * tf.abs(s_l - gt_l), 2.)
                                             + (1. - alpha) * tf.pow(gt_h * tf.abs(s_l - t_l), 2.)
                                             )
                    self.pd_loss = tf.reduce_mean()
            with tf.compat.v1.variable_scope("minimizer"):
                self.two_d_teacher_optim = two_d_optimizer.minimize(self.t_hm_loss + self.t_lm_loss, var_list=self.self.t_var)
                self.two_d_studnet_optim = two_d_optimizer.minimize(self.s_hm_loss + self.s_lm_loss, var_list=self.s_var)
                self.three_d_teachter_optim = three_d_optimizer.minimize(self.t_hm_loss + self.t_lm_loss, var_list=self.self.t_var)
                self.three_d_student_optim = three_d_optimizer.minimize(self.s_hm_loss + self.s_lm_loss, var_list=self.s_var)

        with tf.compat.v1.variable_scope("de_normal"):
            self.t_h = tf.clip_by_value((t_h + 1.) * 127.5, 0., 255.)
            self.t_l = tf.clip_by_value((t_l + 1.) * 127.5, 0., 255.)
            self.s_h = tf.clip_by_value((s_h + 1.) * 127.5, 0., 255.)
            self.s_l = tf.clip_by_value((s_l + 1.) * 127.5, 0., 255.)
            self.gt_h = tf.clip_by_value((gt_h + 1.) * 127.5, 0., 255.)
            self.gt_l = tf.clip_by_value((gt_l + 1.) * 127.5, 0., 255.)

        self.summary_op = tf.compat.v1.summary.merge([
            tf.compat.v1.summary.scalar("loss/t/hm_loss", self.t_hm_loss),
            tf.compat.v1.summary.scalar("loss/t/lm_loss", self.t_lm_loss),
            tf.compat.v1.summary.scalar("loss/s/hm_loss", self.s_hm_loss),
            tf.compat.v1.summary.scalar("loss/s/lm_loss", self.s_lm_loss),
            tf.compat.v1.summary.image("t/h", self.t_h),
            tf.compat.v1.summary.image("t/l", self.t_l),
            tf.compat.v1.summary.image("s/h", self.s_h),
            tf.compat.v1.summary.image("s/l", self.s_l),
            tf.compat.v1.summary.image("gt/h", self.gt_h),
            tf.compat.v1.summary.image("gt/l", self.gt_l),
        ])

    def build_test_model(self):
        print("test_model")

    def test(self):
        self.build_test_model()

    def initial_iterator(self):
        self.sess.run(self.iterator.initializer)

    def preprocessing(self):
        with tf.compat.v1.variable_scope("Pre"):
            two_d_x, two_d_joint = self.data  # [B, 256, 256, 3]  [B, 256, 256, 14]
            two_d_x= self.norm(two_d_x)
            two_d_x = utils.nhwc_to_nchw(two_d_x)
            two_d_joint = utils.nhwc_to_nchw(two_d_joint)
        return two_d_x, two_d_joint

    def norm(self, x):
        return tf.clip_by_value((x/127.5)-1., -1., 1.)

    def load_data(self):
        self.initial_iterator()
        x = self.data
        return x

    def generate(self, image, fixed, root_path=None, path=None, idx=None, save=True):
        x = self.sess.run(self.Gz, {self.z: image})
        if path is None and save:
            path = os.path.join(root_path, '{}_G.mp4'.format(idx))
            utils.save_video(x, fixed, path)
            print("[*] Samples saved: {}".format(path))
        return x

    def mean_rgb(self, x):
        r, g, b = tf.split(x, 3, axis=1)
        mean_r = tf.reduce_mean(r, axis=[2, 3], keep_dims=True)
        mean_g = tf.reduce_mean(g, axis=[2, 3], keep_dims=True)
        mean_b = tf.reduce_mean(b, axis=[2, 3], keep_dims=True)
        r = r - mean_r
        g = g - mean_g
        b = b - mean_b
        return tf.concat([r, g, b], 1), tf.concat([mean_r, mean_g, mean_b], 0)

    def teacher(self, x, j):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            x = utils.nchw_to_nhwc(x)
            batch_out, batch_list = resnet_v1.resnet_v1_50(x, 1000, is_training=True)
            feature = batch_list['resnet_v1_50/block2/unit_4/bottleneck_v1/conv1']

        self.init_fn_1 = slim.assign_from_checkpoint_fn(
            self.pre_dir + '/resnet_v1_50.ckpt', slim.get_model_variables('resnet_v1_50'))

        x = utils.nhwc_to_nchw(feature)
        x, feature = vnect(x, j)
        return x, [batch_list, feature]

    def student(self, x, j):
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
            x = utils.nchw_to_nhwc(x)
            batch_out, batch_list = mobilenet_v2.mobilenet_v2_035(x, 1000, is_training=True)
            # print(batch_list)
            mobilenet_feature = batch_list['layer_12/output']

        self.init_fn = slim.assign_from_checkpoint_fn(
            self.pre_dir + '/mobilenet_v2_035.ckpt', slim.get_model_variables('MobilenetV2'))

        x = utils.nhwc_to_nchw(mobilenet_feature)
        x, var = movnect(x, j)
        return x, [batch_list, var]