import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
import imageio
import os


class plot_callback(tf.keras.callbacks.Callback):

    def __init__(self):
        self.fnames = []
        self.all_prev_vars = None
        self.all_vars = None
        self.weight_update_norms = []

    def on_epoch_end(self, epoch, logs=None):

        if epoch % 5 == 0:
            print(epoch)

            if self.all_prev_vars is None:
                # First weight parameter setting
                self.all_prev_vars = []
                for vars in self.model.trainable_variables:
                    self.all_prev_vars.extend(vars.numpy().ravel())
                self.all_prev_vars = np.array(self.all_prev_vars)

            self.all_current_vars = []
            for vars in self.model.trainable_variables:
                self.all_current_vars.extend(vars.numpy().ravel())
            self.all_current_vars = np.array(self.all_current_vars)

            self.weight_update_norms.append(
                np.linalg.norm(
                    self.all_current_vars - self.all_prev_vars
                )
            )

            self.all_prev_vars = self.all_current_vars

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))

            ax = axes[00]
            xs = np.arange(-5,5,0.01)
            # ys = np.arange(0, 1, 0.01)
            ax.plot(X_train, y_train, 'x', markersize=10, markeredgewidth=3)
            ax.set_ylim([0, 1])
            ax.set_xlim([-5, 5])
            op_dists = self.model(xs)
            mean = op_dists.mean()[:, 0]
            low = (op_dists.mean()-.5*op_dists.variance()**.5)[:, 0]
            high = (op_dists.mean()+.5*op_dists.variance()**.5)[:, 0]
            ax.fill_between(xs, low, high, color='gray', alpha=0.3)
            ax.plot(xs, mean, 'k')
            fname = 'temp/{:05d}.png'.format(epoch)
            ax.set_title('Epoch: {:04d}'.format(epoch))
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')

            ax = axes[1]
            # Don't plot first (zero) norm update
            if len(self.weight_update_norms) > 1:
                ax.plot(np.arange(0, len(self.weight_update_norms))[1:]*5, self.weight_update_norms[1:])
            ax.set_ylabel('L2 norm weight update')
            ax.set_xlim([0, 3000])
            ax.set_xlabel('Epoch')

            plt.savefig(fname)
            self.fnames.append(fname)
            plt.close()

    def on_train_end(self, logs=None):

        with imageio.get_writer('temp/mygif.mp4', mode='I', fps=10) as writer:
            for filename in self.fnames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(self.fnames):
            os.remove(filename)


model = Sequential([
    Dense(100, input_shape=(1,), activation='relu'),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    # Dense(5, activation='relu'),
    Dense(2, activation='linear'),
    tfpl.DistributionLambda(  # Trying to clip by value to avoid nans
        lambda t: tfd.Independent(tfd.TruncatedNormal(
            # loc=t[..., :1],
            loc=tf.math.sigmoid(t[..., :1]),
            # loc=1.01*tf.math.sigmoid(t[..., :1]) - 0.01/2,
            # scale=tf.clip_by_value(tf.math.softplus(t[..., 1:]), 0.0001, np.inf),
            scale=tf.math.softplus(t[..., 1:]) + 0.0001,
            # scale=tf.math.softplus(t[..., 1:]),
            low=0.,
            high=1.)
        ))
        # lambda t: tfd.Independent(tfd.Normal(
        #     loc=t[..., :1],
        #     scale=tf.clip_by_value(tf.math.softplus(t[..., 1:]), 5e-4, np.inf))
        # ))
])


def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


X_train = tf.random.normal((40, 1), stddev=1)
y_train = X_train**3 + 2*X_train**2 + 4
y_train += tf.random.normal((40, 1), stddev=0.1)
y_train = (y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train))

model.compile(loss=nll, optimizer=tf.keras.optimizers.Adam(lr=5e-4))
history = model.fit(X_train, y_train, callbacks=[plot_callback()], epochs=3000, verbose=0)
plt.plot(history.history['loss'])

if True:
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(-3,3,0.01)[:, np.newaxis]  # Each point is a new batch
    ys = np.arange(0, 1, 0.01)[:, np.newaxis]
    ax.plot(X_train, y_train, 'x', markersize=10, markeredgewidth=3)
    ax.set_ylim([0, 1])
    op_dists = model(xs)
    for X in X_train:
        train_dist = model(X)
        probs = train_dist.prob([ys])
        probs = train_dist.prob(ys)
        probs /= 10*np.max(probs)
        ax.plot(X-probs, ys, 'k')
        ax.axvline(X, color='k', alpha=0.3)
    # for conf inf plotting reshape to 1D
    xs = xs[:, 0]
    mean = op_dists.mean()[:, 0]
    low = (op_dists.mean()-.5*op_dists.variance()**.5)[:, 0]
    high = (op_dists.mean()+.5*op_dists.variance()**.5)[:, 0]
    ax.fill_between(xs, low, high, color='gray', alpha=0.3)
    ax.plot(xs, op_dists.mean(), 'k')
    fig
