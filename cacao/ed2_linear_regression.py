from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tensorflow_probability import edward2 as ed

import tensorflow as tf

class LinearModel(object):
    def __initi__(self):
        self.hparams = hparams
        self.weight_dict = {'w':None, 'b':None}
    
    def fit(self, input_fn):

        def linear_model(X): # (unmodeled) data
            w = ed.Normal(loc=tf.zeros([X.shape[1]]),
                scale=tf.ones([X.shape[1]]),
                name="w")  # parameter
            b = ed.Normal(loc=tf.zeros([]),
                scale=tf.ones([]), 
                name="b")  # parameter
            y = ed.Normal(loc=tf.tensordot(X, w, axes=1) + b, scale=1.0, 
                name="y")  # (modeled) data
            return y   

        def variational_model(qw_mean, qw_stddv, qb_mean, qb_stddv):
            qw = ed.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
            qb = ed.Normal(loc=qb_mean, scale=qb_stddv, name="qb")          
            return qw, qb

        x_tensor = tf.convert_to_tensor(x_train, tf.float32)
        y_tensor = tf.convert_to_tensor(y_train, tf.float32)
        
        # make dataset
        dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
        shuffle = dataset.shuffle(hparams.shuffle_iterations)
        batches = shuffle.repeat().batch(hparams.batch_size)
        iterator = batches.make_one_shot_iterator()
        features, labels = iterator.get_next()

        log_q = ed.make_log_joint_fn(variational_model)

        def target_q(qw, qb):
            return log_q(qw_mean=qw_mean, qw_stddv=qw_stddv, qb_mean=qb_mean, qb_stddv=qb_stddv, qw=qw, qb=qb)

        qw_mean = tf.Variable(tf.zeros([int(features.shape[1])]), dtype=tf.float32)
        qb_mean = tf.Variable(tf.zeros([]), dtype=tf.float32)
        qw_stddv = tf.nn.softplus(tf.Variable(tf.ones([int(features.shape[1])]), dtype=tf.float32))
        qb_stddv = tf.nn.softplus(tf.Variable(tf.ones([]), dtype=tf.float32))

        qw, qb = variational_model(qw_mean=qw_mean, qw_stddv=qw_stddv,
                                qb_mean=qb_mean, qb_stddv=qb_stddv)

        log_joint = ed.make_log_joint_fn(linear_model)

        def target(qw, qb):
            """Unnormalized target density as a function of the parameters."""
            return log_joint(w=qw, b=qb, X=features, y=labels)

            energy = target(qw, qb) 
            entropy = -target_q(qw, qb) / hparams.batch_size
            elbo = energy + entropy

            optimizer = tf.train.AdamOptimizer(learning_rate = hparams.learning_rate)
            train = optimizer.minimize(-elbo)

            init = tf.global_variables_initializer()

            t = []
            weights_dict = {}
            num_steps = int(x_train.shape[0] / hparams.batch_size)

        with tf.Session() as sess:
            sess.run(init)

            for step in range(num_steps):
                _ = sess.run([train])
                if step % 100 == 0:
                    t.append(sess.run([elbo]))
            
            weights_dict['w'], weights_dict['b'] = sess.run([qw.distribution.sample(1000), qb.distribution.sample(1000)])
        
        return weights_dict['w'], weights_dict['b']

    def evaluate(self, input_fn):

        features, labels = tf.convert_to_tensor(x_test, tf.float32), tf.convert_to_tensor(y_test, tf.float32)

        w_ = tf.reduce_mean(qw_, 0)
        b_ = tf.reduce_mean(qb_, 0)

        pred_tensor = tf.cast((tf.tensordot(features, w_, axes=1) + b_), tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((labels, pred_tensor))
        batches = dataset.repeat().batch(hparams.shuffle_iterations)
        iterator = batches.make_one_shot_iterator()
        labels, predictions = iterator.get_next()

        rmse, rmse_update_op = tf.metrics.root_mean_squared_error(tf.cast(labels,tf.float32), predictions)

        n_batches = int(x_test.shape[0] / hparams.batch_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for _i in range(n_batches):
                rmse_ = sess.run([rmse_update_op])

            metrics = {}
            metrics["rmse"] = sess.run([rmse])
            return metrics
