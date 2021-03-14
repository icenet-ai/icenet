import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import utils
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio
from tqdm import tqdm

all_network_dataset_video = True

if all_network_dataset_video:

    network_dataset = 'dataset1'
    dataset_folder = os.path.join('data', 'network_datasets', network_dataset, 'obs')
    vars = os.listdir(dataset_folder)

    video_folder = os.path.join('videos', 'network_dataset_videos', network_dataset)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    video_path = os.path.join(video_folder, 'vid.mp4')
    fps = 60

    nrows = 2
    ncols = int(np.ceil(len(vars)/2))
    vars
    fnames = os.listdir(os.path.join(dataset_folder, 'siconca', 'abs'))
    data_formats = [os.listdir(os.path.join(dataset_folder, var))[0] for var in vars]

    def make_frame(fname):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2*ncols, 2*nrows))
        axes = axes.ravel()
        for var, fmt, ax in zip(vars, data_formats, axes):
            fpath = os.path.join(dataset_folder, var, fmt, fname)
            if not os.path.exists(fpath):
                arr = np.zeros((423, 432))
            else:
                arr = np.load(fpath)
            ax.imshow(arr)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.set_title(var)

        plt.suptitle(fname)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()
        return image

    imageio.mimsave(video_path,
                    [make_frame(fname) for fname in tqdm(fnames)],
                    fps=fps)
