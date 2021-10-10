import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def evaluate(exp, prediction, num, image, title):
    """
    Create a gif file of evaluation and saves it to path
    :param exp: path to experiment
    :param prediction: prediction name
    :param num: number of groundtruth-prediction pair
    :param path: name of image
    :param title: title of the image
    :return:
    """
    pred_path = '../experiments/' + exp + '/predictions/' + prediction + '/prediction_' + num + '.npy'
    gt_path = '../experiments/' + exp + '/predictions/' + prediction + '/groundtruth_' + num + '.npy'
    gt = np.load(gt_path)
    pred = np.load(pred_path)
    err = np.abs(gt-pred)

    fig = plt.figure(figsize=(40, 10))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    f = 30
    fig.suptitle(title, fontsize=f*1.2)
    ax1.set_title('Ground truth', fontsize=f)
    ax2.set_title('Prediction', fontsize=f)
    ax3.set_title('Error', fontsize=f)
    labels = [0, 5, 10, 15, 20, 25, 30]
    ax1.set_xticklabels(labels, fontsize=f)
    ax1.set_yticklabels(labels, fontsize=f)
    ax2.set_xticklabels(labels, fontsize=f)
    ax2.set_yticklabels(labels, fontsize=f)
    ax3.set_xticklabels(labels, fontsize=f)
    ax3.set_yticklabels(labels, fontsize=f)

    quad1 = ax1.pcolormesh(gt[:, :, 0])
    quad2 = ax2.pcolormesh(pred[:, :, 0])
    quad3 = ax3.pcolormesh(err[:, :, 0])
    cb1 = fig.colorbar(quad1, ax=ax1)
    cb2 = fig.colorbar(quad2, ax=ax2)
    cb3 = fig.colorbar(quad3, ax=ax3)
    cb1.ax.tick_params(labelsize=f * 0.7)
    cb2.ax.tick_params(labelsize=f * 0.7)
    cb3.ax.tick_params(labelsize=f * 0.7)

    def init_animation():
        quad1.set_array([])
        quad2.set_array([])
        quad3.set_array([])
        return quad1, quad2, quad3,

    def animate(i):
        quad1.set_array(gt[:, :, i])
        quad2.set_array(pred[:, :, i])
        quad3.set_array(err[:, :, i])
        error = np.sum((gt[:, :, i] - pred[:, :, i]) ** 2) / np.sum(gt[:, :, i] ** 2)
        ax3.set_title('Error: ' + str(error)[0:5], fontsize=f)
        return quad1,

    writergif = animation.PillowWriter(fps=1)
    anim = animation.FuncAnimation(fig, animate, init_func=init_animation, frames=40, interval=200, blit=True)
    name = image + '.gif'
    anim.save(name, dpi=50, writer=writergif)


evaluate('heat_3d', 'pred_gt', '0001', 'heat_3d_eval_0001', 'Heat equation, 3d FNO, sample 0001')
