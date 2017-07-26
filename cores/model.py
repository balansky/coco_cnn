from inception import inception_model
import tensorflow as tf


TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'

class InceptionMultiLabels(object):

    def __init__(self, images, cat_num, mode):
        self.cat_num = cat_num
        if mode in (TRAIN, EVAL):
            self.logits, self.end_points = inception_model.inference(images, self.cat_num, for_training=True)
        else:
            self.logits, self.end_points = inception_model.inference(images, self.cat_num, for_training=False)


    # def _build_graph(self, images, is_training=True):
    #     logits, end_points = inception_model.inference(images, self.cat_num + 1, for_training=is_training)
    #     inception_embeddings = end_points['prelogits']
    #     sigmoids = tf.nn.sigmoid(logits, name='sigmoid_result')
    #     return sigmoids, logits, inception_embeddings

    #
    # def _restore_graph(self, sess, checkpoint_dir):
    #     global_step = None
    #     variable_averages = tf.train.ExponentialMovingAverage(0.9999)
    #     variables_to_restore = variable_averages.variables_to_restore()
    #     saver = tf.train.Saver(variables_to_restore)
    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    #         print('Successfully loaded model from %s at step=%s.' % (ckpt.model_checkpoint_path, global_step))
    #     else:
    #         print('No checkpoint file found at %s') % checkpoint_dir
    #     return global_step

    def train_fn(self, images, labels, learning_rate, fully_train=True):
        idx_table = tf.contrib.lookup.index_table_from_tensor(mapping=labels, num_oov_buckets=1, default_value=-1)

        logits, self.end_points = inception_model.inference(images, self.cat_num, for_training=fully_train)
        global_step = tf.Variable(0, trainable=False)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        lr = tf.train.exponential_decay(learning_rate, global_step, 3000, 0.96, staircase=True)
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step=global_step)
        return train_step, loss


    def predict_fn(self):
        pass

    def model_fn(self, mode, learning_rate=0.01):
        if mode in (PREDICT, EVAL):
            pass

    def inference(self, sigmoids):
        return tf.round(sigmoids)

    def eval(self, sigmoids, labels):
        correct_prediction = tf.reduce_mean(tf.equal(tf.round(sigmoids), labels))
        return correct_prediction
