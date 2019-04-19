''' Functions for tensorflow '''

import tensorflow as tf

def negative_MLS(X, Y, sigma_sq_X, sigma_sq_Y, mean=False):
    with tf.name_scope('negative_MLS'):
        if mean:
            D = X.shape[1].value

            Y = tf.transpose(Y)
            XX = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
            YY = tf.reduce_sum(tf.square(Y), 0, keep_dims=True)
            XY = tf.matmul(X, Y)
            diffs = XX + YY - 2*XY

            sigma_sq_Y = tf.transpose(sigma_sq_Y)
            sigma_sq_X = tf.reduce_mean(sigma_sq_X, axis=1, keep_dims=True)
            sigma_sq_Y = tf.reduce_mean(sigma_sq_Y, axis=0, keep_dims=True)
            sigma_sq_fuse = sigma_sq_X + sigma_sq_Y

            diffs = diffs / (1e-8 + sigma_sq_fuse) + D * tf.log(sigma_sq_fuse)

            return diffs
        else:
            D = X.shape[1].value
            X = tf.reshape(X, [-1, 1, D])
            Y = tf.reshape(Y, [1, -1, D])
            sigma_sq_X = tf.reshape(sigma_sq_X, [-1, 1, D])
            sigma_sq_Y = tf.reshape(sigma_sq_Y, [1, -1, D])
            sigma_sq_fuse = sigma_sq_X + sigma_sq_Y
            diffs = tf.square(X-Y) / (1e-10 + sigma_sq_fuse) + tf.log(sigma_sq_fuse)
            return tf.reduce_sum(diffs, axis=2)

def mutual_likelihood_score_loss(labels, mu, log_sigma_sq):

    with tf.name_scope('MLS_Loss'):

        batch_size = tf.shape(mu)[0]

        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)

        sigma_sq = tf.exp(log_sigma_sq)
        loss_mat = negative_MLS(mu, mu, sigma_sq, sigma_sq)
        
        label_mat = tf.equal(labels[:,None], labels[None,:])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)

        loss_pos = tf.boolean_mask(loss_mat, label_mask_pos) 
    
        return tf.reduce_mean(loss_pos)
