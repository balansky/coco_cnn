from cores import model
import tensorflow as tf
from inception.slim import slim


BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

class FinetuneIncpt(model.MultiLabelTrainer):

    def _cross_entropy(self, batch_images, batch_labels, for_training=False):
        if not for_training:
            tf.get_variable_scope().reuse_variables()
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
        }
        # Set weight_decay for weights in Conv and FC layers.
        with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
            with slim.arg_scope([slim.ops.conv2d],
                                stddev=0.1,
                                activation=tf.nn.relu,
                                batch_norm_params=batch_norm_params):
                logits, endpoints = slim.inception.inception_v3(
                    batch_images,
                    dropout_keep_prob=0.8,
                    num_classes=1001,
                    is_training=for_training,
                    restore_logits=True,
                    scope=None)
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=batch_labels, logits=logits, pos_weight=self.cls_num/2)
        total_loss = tf.reduce_mean(cross_entropy)
        return total_loss, logits


def main():
    with tf.Session() as sess:
        trainer = FinetuneIncpt(data_dir, config_dir, sup_cats)
        batch_inputs, batch_labels = trainer.input_fn('train', train_batch_size, num_epochs, num_threads)
        batch_eval_inputs, batch_eval_labels = trainer.input_fn('val', train_batch_size, num_epochs, num_threads)
        train_op, global_step = trainer.train_fn(batch_inputs, batch_labels, learning_rate, decay_frequency, decay_rate)
        eval_loss, eval_acc = trainer.eval_fn(batch_eval_inputs, batch_eval_labels)
        train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.train.Saver(train_variables)
        saver.restore(sess, tf.train.latest_checkpoint(pretrained_model_path))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(300000):
            gs, _ = sess.run([global_step, train_op])
            if i % 100:
                saver.save(sess, save_dir, global_step=gs)
            if i % 10:
                el, ea = sess.run([eval_loss, eval_acc])
                print("At Step %s, Evaluation Loss : %s, Evaluation Accuracy: %s " % (i, el, ea))
        coord.request_stop()
        coord.join(threads)
        print('Finished')



if __name__== "__main__":
    data_dir = "/home/andy/Data/coco/tfrecords"
    config_dir = "configs"
    sup_cats = "food"
    save_dir = "ckpts"
    train_batch_size = 12
    num_epochs = None
    num_threads = 8
    learning_rate = 0.0001
    decay_frequency = 10000
    decay_rate = 0.95
    pretrained_model_path = "/home/andy/Data/models/inception-v3/"
    main()