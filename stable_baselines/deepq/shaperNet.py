import tensorflow as tf
import numpy as np


class Shaper:
    def __init__(self, ob_space, learning_rate=3e-4, batch_size=64):
        # with tf.device('/gpu:0'):
        self.ob_space = ob_space.shape
        self.learning_rate = learning_rate
        self.batchsize = batch_size
        self.sess = tf.Session()
        self.dense1 = None
        self.state_pairs_logit = None
        # dimension is obs + next_obs
        flatten_obs_dim = np.prod(np.array(self.ob_space)) * 2

        with tf.variable_scope("input", reuse=None):
            self.state_pairs_mb = tf.placeholder(
                tf.float32, shape=[None, flatten_obs_dim], name="statePairs"
            )
            self.state_pairs_label = tf.placeholder(tf.float32, shape=[None, 2])
            self.original_reward = tf.placeholder(tf.float32, shape=[None])
            self.shape_vec = tf.placeholder(tf.float32, shape=[None, 2])

        with tf.variable_scope("model", reuse=None):
            self.dense0 = tf.layers.dense(
                inputs=self.state_pairs_mb, units=64, activation=tf.nn.relu
            )
            self.dense1 = tf.layers.dense(
                inputs=self.dense0, units=64, activation=tf.nn.relu
            )
            self.state_pairs_logit = tf.layers.dense(
                inputs=self.dense1, units=2, activation=None
            )

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.state_pairs_label, logits=self.state_pairs_logit
        )
        self.loss = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate
        ).minimize(self.loss)

        self.shaped_reward_vec = tf.dtypes.cast(
            self.shape_vec[:, 0] > self.shape_vec[:, 1], tf.float32
        )

        tf.global_variables_initializer().run(session=self.sess)

    def predict(self, x):
        return self.sess.run(self.state_pairs_logit, {self.state_pairs_mb: x})

    def optimize(self, x, y):
        loss, optimizer, prediction = self.sess.run(
            [self.loss, self.optimizer, self.state_pairs_logit],
            feed_dict={self.state_pairs_mb: x, self.state_pairs_label: y},
        )
        return loss, optimizer, prediction

    def prediction(self, x):
        return self.sess.run(self.state_pairs_logit, {self.state_pairs_mb: x})

    def shape_reward(self, original_reward_vec, shaper_output):
        return self.sess.run(
            self.shaped_reward_vec,
            {self.shape_vec: shaper_output, self.original_reward: original_reward_vec},
        )
