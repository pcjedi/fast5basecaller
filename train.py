import tensorflow as tf
from tqdm import tqdm
from fast5_input import fast5batches
from model import x, sequence_length, num_features, batch_size, segment_length, logits

with tf.name_scope('labels'):
    indices = tf.placeholder(tf.int64, name='indices')
    y = tf.placeholder(tf.int32, name='values')
    dense_shape = (batch_size, segment_length)

    labels = tf.SparseTensor(indices = indices, 
                             values = y, 
                             dense_shape = dense_shape
                            )
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.ctc_loss(labels = labels, 
                       inputs = logits, 
                       sequence_length = sequence_length, 
                       time_major=False
                      )
    )
    
global_step = tf.get_variable('global_step', trainable=False, shape=(),
                              dtype=tf.int32,
                              initializer=tf.zeros_initializer())

learning_rate = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

step = optimizer.minimize(loss, global_step = global_step)

with tf.name_scope('error'):
    logits_transposed = tf.transpose(logits, perm=[1, 0, 2])
    decoded, neg_sum_logits = tf.nn.ctc_beam_search_decoder(
                logits_transposed,
                sequence_length,
                merge_repeated=False,
                top_paths=1,
                beam_width=30)
    first_path = decoded[0]
    edit_d = tf.edit_distance(tf.to_int32(first_path), labels, normalize=True)
    error = tf.reduce_mean(edit_d, axis=0)
    
save_path = 'log/save'

summary_train = tf.summary.merge([tf.summary.scalar('learning_rate', learning_rate), 
                                  tf.summary.scalar('loss', loss)])
summary_test = tf.summary.merge([tf.summary.scalar('Error_rate', error)])

f5b = fast5batches(batch_size=batch_size, 
                   segment_length=segment_length, 
                   fast5dir='../chiron-otrain/pass', 
                   training=True, 
                   test_ratio=.2,
                   overlap=30)

f5b.next_batch(test=False, fill=True)
f5b.next_batch(test=True, fill=True)

name = 'withalllast_exp-2_1'
summary_path = 'logs/summary/' + name
lr_val = 10**-2
err_val_low = None

with tf.Session() as sess:
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
    try:
        saver.restore(sess, './logs/save/{}.ckpt'.format(name))
        
        X, ln, Y, fi = f5b.next_batch(test=True)
        feed_dict = {x: X, 
                     sequence_length: ln, 
                     indices: Y.indices, 
                     y: Y.values}
        err_val_low, gs_val = sess.run([error, global_step], feed_dict = feed_dict)
        desc = '{:.2%} lowest error rate at {}, training'.format(err_val_low, gs_val)
    except ValueError:
        print('initialize variables')
        sess.run(tf.global_variables_initializer())
    with tqdm() as bar:
        while True:
            X, ln, Y, fi = f5b.next_batch(test=False)
            feed_dict = {x: X, 
                         sequence_length: ln, 
                         indices: Y.indices, 
                         y: Y.values,
                         learning_rate: lr_val}
            summary, _, gs_val, loss_val = sess.run([summary_train, step, global_step, loss], feed_dict = feed_dict)
            summary_writer.add_summary(summary, global_step=gs_val)
            summary_writer.flush()
            bar.n = gs_val
            bar.refresh()
            if gs_val%10==0:
                X, ln, Y, fi = f5b.next_batch(test=True)
                feed_dict = {x: X, 
                             sequence_length: ln, 
                             indices: Y.indices, 
                             y: Y.values,}
                summary, err_val, gs_val = sess.run([summary_test, error, global_step], feed_dict = feed_dict)
                summary_writer.add_summary(summary, global_step=gs_val)
                summary_writer.flush()
                if err_val_low is None or err_val<err_val_low:
                    err_val_low = err_val
                    bar.desc = '{:.2%} lowest error rate at {}, training'.format(err_val, gs_val)
                    saver.save(sess, './logs/save/{}.ckpt'.format(name))