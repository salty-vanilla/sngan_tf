import tensorflow as tf
import numpy as np
import os
import csv
import time
from PIL import Image


class GAN:
    def __init__(self, generator,
                 discriminator,
                 metrics='JSD',
                 lr_d=1e-4,
                 lr_g=1e-4,
                 eps=1e-12,
                 is_training=True):
        self.discriminator = discriminator
        self.generator = generator
        self.image_shape = self.discriminator.input_shape
        self.noise_dim = self.generator.noise_dim
        self.image = tf.placeholder(tf.float32, [None] + list(self.image_shape), name='x')
        self.noise = tf.placeholder(tf.float32, [None, self.noise_dim], name='z')

        self.generate = self.generator(self.noise, reuse=False, is_training=True)
        self.generate_ = self.generator(self.noise, reuse=True, is_training=False)

        self.discriminate_real = self.discriminator(self.image, reuse=False, is_training=True)
        self.discriminate_fake = self.discriminator(self.generate, reuse=True, is_training=True)

        self.discriminate_fake_ = self.discriminator(self.generate, reuse=True, is_training=False)

        with tf.name_scope('loss'):
            if metrics == 'JSD':
                real_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.discriminate_real),
                                                            logits=self.discriminate_real))
                fake_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.discriminate_fake),
                                                            logits=self.discriminate_fake))

                self.loss_d = real_loss + fake_loss

                self.loss_g = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.discriminate_fake_),
                                                            logits=self.discriminate_fake_))

            elif metrics == 'WD':
                self.loss_g = -tf.reduce_mean(self.discriminate_fake)
                self.loss_d = -(tf.reduce_mean(self.discriminate_real)
                                - tf.reduce_mean(self.discriminate_fake))
            else:
                raise NotImplementedError

        # Optimizer
        if is_training:
            with tf.name_scope('Optimizer'):
                if metrics == 'JSD':
                    self.opt_d = tf.train.AdamOptimizer(learning_rate=lr_d, beta1=0.5, beta2=0.99) \
                        .minimize(self.loss_d,
                                  var_list=self.discriminator.vars)
                    self.opt_g = tf.train.AdamOptimizer(learning_rate=lr_g, beta1=0.5, beta2=0.99) \
                        .minimize(self.loss_g,
                                  var_list=self.generator.vars)
                elif metrics == 'WD':
                    self.opt_d = tf.train.AdamOptimizer(learning_rate=lr_d, beta1=0.5, beta2=0.99) \
                        .minimize(self.loss_d,
                                  var_list=self.discriminator.vars)
                    self.opt_g = tf.train.AdamOptimizer(learning_rate=lr_g, beta1=0.5, beta2=0.99) \
                        .minimize(self.loss_g,
                                  var_list=self.generator.vars)
                else:
                    raise NotImplementedError
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.model_dir = None
        self.sess.run(tf.global_variables_initializer())
        self.is_training = False

        self.tb_writer = tf.summary.FileWriter('../logs', graph=self.sess.graph)

    def fit(self, image_sampler,
            noise_sampler,
            nb_epoch=1000,
            visualize_steps=1,
            save_steps=1,
            result_dir='result',
            model_dir='model'):
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample
        self.model_dir = model_dir

        # prepare for csv
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        f = open(os.path.join(result_dir, 'learning_log.csv'), 'w')
        writer = csv.writer(f, lineterminator='\n')

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1

        # for display and csv
        loss_g = 0
        writer.writerow(['loss_d', 'loss_g'])

        # fit loop
        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            start = time.time()
            for iter_ in range(1, steps_per_epoch + 1):
                image_batch = image_sampler()
                noise_batch = noise_sampler(image_batch.shape[0], self.noise_dim)
                _, loss_d, = self.sess.run([self.opt_d, self.loss_d],
                                           feed_dict={self.image: image_batch,
                                                      self.noise: noise_batch})
                _, loss_g = self.sess.run([self.opt_g, self.loss_g],
                                          feed_dict={self.noise: noise_batch})

                print('iter : {} / {}  {:.1f}[s]  loss_d : {:.4f}  loss_g : {:.4f}  \r'
                      .format(iter_, steps_per_epoch, time.time() - start,
                              loss_d, loss_g), end='')
                writer.writerow([loss_d, loss_g])

            if epoch % visualize_steps == 0:
                noise_batch = noise_sampler(batch_size, self.noise_dim)
                self.visualize(os.path.join(result_dir, 'epoch_{}'.format(epoch)),
                               noise_batch, image_sampler.data_to_image)
            if epoch % save_steps == 0:
                self.save(epoch)
        print('\nTraining is done ...\n')

    def restore(self, file_path, mode='both'):
        assert mode in ['both', 'discriminator']
        reader = tf.train.NewCheckpointReader(file_path)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0])
                            for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        var_dict = dict(zip(map(lambda x:
                                x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                if mode == 'discriminator' and 'discriminator' not in var_name:
                    continue
                current_var = var_dict[saved_var_name]
                var_shape = current_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(current_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(self.sess, file_path)

    def visualize(self, dst_dir, noise_batch, convert_function):
        generated_data = self.predict_on_batch(noise_batch)
        generated_images = convert_function(generated_data)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for i, image in enumerate(generated_images):
            if image.shape[2] == 1:
                image = image.reshape(image.shape[:2])
            dst_path = os.path.join(dst_dir, "{}.png".format(i))
            pil_image = Image.fromarray(np.uint8(image))
            pil_image.save(dst_path)

    def save(self, epoch):
        dst_dir = os.path.join(self.model_dir, "epoch_{}".format(epoch))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        return self.saver.save(self.sess, save_path=os.path.join(dst_dir, 'model.ckpt'))

    def predict(self, x, batch_size=16):
        outputs = np.empty([0] + list(self.image_shape))
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            o = self.predict_on_batch(x_batch)
            outputs = np.append(outputs, o, axis=0)
        return outputs

    def predict_on_batch(self, x):
        return self.sess.run(self.generate_,
                             feed_dict={self.noise: x})