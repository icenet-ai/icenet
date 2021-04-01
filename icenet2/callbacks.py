import os
import sys
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))  # if using jupyter kernel
import config
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
import wandb
import matplotlib.pyplot as plt

###############################################################################
############### CALLBACKS
###############################################################################


class IceNetPreTrainingEvaluator(tf.keras.callbacks.Callback):
    """
    Custom tf.keras callback to update the `logs` dict used by all other callbacks
    with the validation set metrics. The callback is executed every
    `validation_frequency` batches.

    This can be used in conjuction with the BatchwiseModelCheckpoint callback to
    perform a model checkpoint based on validation data every N batches - ensure
    the `save_frequency` input to BatchwiseModelCheckpoint is also set to
    `validation_frequency`.

    Also ensure that the callbacks list past to Model.fit() contains this
    callback before any other callbacks that need the validation metrics.

    Also use Weights and Biases to log the training and validation metrics.
    """

    def __init__(self, validation_frequency, val_dataloader, sample_at_zero=False):
        self.validation_frequency = validation_frequency
        self.sample_at_zero = sample_at_zero
        self.val_dataloader = val_dataloader

    def on_train_batch_end(self, batch, logs=None):

        if (batch == 0 and self.sample_at_zero) or (batch + 1) % self.validation_frequency == 0:
            val_logs = self.model.evaluate(self.val_dataloader, verbose=0, return_dict=True)
            val_logs = {'val_' + name: val for name, val in val_logs.items()}
            logs.update(val_logs)
            [print('\n' + k + ' {:.2f}'.format(v)) for k, v in logs.items()]
            print('\n')


class BatchwiseWandbLogger(tf.keras.callbacks.Callback):
    """
    Docstring TODO
    """

    def __init__(self, batch_frequency, log_metrics=True,
                 log_weights=False, log_figure=False, dataloader=None,
                 sample_at_zero=False):
        self.batch_frequency = batch_frequency
        self.log_metrics = log_metrics
        self.log_weights = log_weights
        self.log_figure = log_figure
        self.dataloader = dataloader
        self.sample_at_zero = sample_at_zero

        self.log_figure_init_date = datetime(2012, 10, 1)
        self.log_figure_lead_days = 31

        if log_figure:
            self.land_mask = np.load(os.path.join('data', 'nh', 'masks',
                                                  config.fnames['land_mask']))

    def on_train_batch_end(self, batch, logs=None):

        if (batch == 0 and self.sample_at_zero) or (batch + 1) % self.batch_frequency == 0:
            if self.log_metrics:
                wandb.log(logs)

            if self.log_figure:
                X, y = self.dataloader.data_generation([('nh', self.log_figure_init_date)])
                pred = self.model.predict(X)
                mask = y[:, :, :, :, 1] == 0
                pred[mask] = 0

                pred_case_study = 100*pred[0, :, :, self.log_figure_lead_days-1]
                y_true_case_study = 100*y[0, :, :, self.log_figure_lead_days-1, 0]

                err = pred_case_study - y_true_case_study

                mae = np.mean(np.abs(err[~mask[0, :, :, self.log_figure_lead_days-1]]))

                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

                ax = axes[0]
                im = ax.imshow(y_true_case_study[75:325, 75:325], cmap='Blues_r', clim=(0, 100))
                ax.contour(self.land_mask[75:325, 75:325], levels=[.5], colors='k')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax)
                ax.set_title('True map')

                ax = axes[1]
                im = ax.imshow(pred_case_study[75:325, 75:325], cmap='Blues_r', clim=(0, 100))
                ax.contour(self.land_mask[75:325, 75:325], levels=[.5], colors='k')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax)
                ax.set_title('IceNet2 prediction, MAE: {:.2f}%'.format(mae))

                ax = axes[2]
                im = ax.imshow(err[75:325, 75:325], cmap='seismic', clim=(-100, 100))
                ax.contour(self.land_mask[75:325, 75:325], levels=[.5], colors='k')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax)
                ax.set_title('Prediction minus true')

                for ax in axes:
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                plt.tight_layout()

                wandb.log({'val_case_study_err_map': fig})
                plt.close()

            if self.log_weights:
                # Taken from
                metrics = {}
                for layer in self.model.layers:
                    weights = layer.get_weights()
                    if len(weights) == 1:
                        metrics["parameters/" + layer.name +
                                ".weights"] = wandb.Histogram(weights[0])
                    elif len(weights) == 2:
                        metrics["parameters/" + layer.name +
                                ".weights"] = wandb.Histogram(weights[0])
                        metrics["parameters/" + layer.name +
                                ".bias"] = wandb.Histogram(weights[1])
                wandb.log(metrics)


class WandbGradientUpdateLogger(tf.keras.callbacks.Callback):
    """
    Docstring TODO
    """

    def __init__(self, log_layerwise_weight_hists=False,
                 log_weight_update_hists=True, log_weight_update_norm=True):
        self.log_layerwise_weight_hists = log_layerwise_weight_hists
        self.log_weight_update_hists = log_weight_update_hists
        self.log_weight_update_norm = log_weight_update_norm

    def get_weights(self):

        # Store weights
        weights_dict = {}
        for layer in self.model.layers:
            weights = layer.get_weights()
            if len(weights) == 1:
                weights_dict["parameters/" + layer.name + ".weights"] = weights[0].ravel()
            elif len(weights) == 2:
                weights_dict["parameters/" + layer.name + ".weights"] = weights[0].ravel()
                weights_dict["parameters/" + layer.name + ".bias"] = weights[1].ravel()

        return weights_dict

    def on_train_batch_begin(self, batch, logs=None):

        # Set previous weights dict before first batch of the epoch
        if batch == 0:
            self.weights_dict_prev_batch = self.get_weights()

    def on_train_batch_end(self, batch, logs=None):

        '''Compute change in weights from before gradient update'''

        weights_dict_this_batch = self.get_weights()

        weight_updates_this_batch = {}
        for key, vals in weights_dict_this_batch.items():
            weight_updates_this_batch[key + '.updates'] = weights_dict_this_batch[key] - self.weights_dict_prev_batch[key]

        # Log layerwise gradient updates to wandb
        if self.log_weight_update_hists:
            weight_update_hist_dict = {k: wandb.Histogram(vals) for k, vals in weight_updates_this_batch.items()}
            wandb.log(weight_update_hist_dict)

        if self.log_layerwise_weight_hists:
            wandb.log(weights_dict_this_batch)

        # Log ALL gradient updates to wandb
        all_weight_updates = [val for vals in weight_updates_this_batch.values() for val in vals]
        if self.log_weight_update_hists:
            all_weight_updates_hist_dict = {'all_weight_updates_hist': wandb.Histogram(all_weight_updates)}
            wandb.log(all_weight_updates_hist_dict)
        if self.log_weight_update_norm:
            all_weight_updates_norm_dict = {'all_weight_updates_norm': np.linalg.norm(all_weight_updates)}
            wandb.log(all_weight_updates_norm_dict)

        # Reset previous weights dict
        self.weights_dict_prev_batch = weights_dict_this_batch


class BatchwiseModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Docstring TODO
    """

    def __init__(self, save_frequency, model_path, mode, monitor, prev_best=None, sample_at_zero=False):
        self.save_frequency = save_frequency
        self.model_path = model_path
        self.mode = mode
        self.monitor = monitor
        self.sample_at_zero = sample_at_zero

        if prev_best is not None:
            self.best = prev_best

        else:
            if self.mode == 'max':
                self.best = -np.Inf
            elif self.mode == 'min':
                self.best = np.Inf

    def on_train_batch_end(self, batch, logs=None):

        if (batch == 0 and self.sample_at_zero) or (batch + 1) % self.save_frequency == 0:
            if self.mode == 'max' and logs[self.monitor] > self.best:
                save = True

            elif self.mode == 'min' and logs[self.monitor] < self.best:
                save = True

            else:
                save = False

            if save:
                print('\n{} improved from {:.3f} to {:.3f}. Saving model to {}.\n'.
                      format(self.monitor, self.best, logs[self.monitor], self.model_path))

                self.best = logs[self.monitor]

                self.model.save(self.model_path, overwrite=True)
            else:
                print('\n{}={:.3f} did not improve from {:.3f}\n'.format(self.monitor, logs[self.monitor], self.best))
