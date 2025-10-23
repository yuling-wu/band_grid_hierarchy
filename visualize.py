# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

import scipy
import scipy.stats
from imageio import imsave
import cv2


def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap='jet', smooth=True):
    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')  # ignore divide by zero err
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(activations, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb(im, cmap, smooth) for im in activations[:n_plots]]
    if width > n_plots:
        width = n_plots
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig


def map2pi(d):
    d = np.where(d<-np.pi, d+np.pi*2, d)
    d = np.where(d>np.pi, d-np.pi*2, d)
    return d


def compute_ratemaps(model, trajectory_generator, options, res=20, n_avg=None, Ng=512, idxs=None, compute_RNN_hierarchy=None):
    '''Compute spatial firing fields and orientation tuning'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length
    if options.Ng < Ng:
        Ng = options.Ng
    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ng])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Ng, res, res]) 
    counts  = np.zeros([res, res])

    activations_theta = np.zeros([Ng, res*2])
    bins = np.linspace(-np.pi, np.pi, res*2, endpoint=False)
    counts_theta  = np.zeros([res*2])

    for index in range(n_avg):
        inputs, pos_batch, pc_outputs, theta_batch = trajectory_generator.get_test_batch()
        if options.RNN_type == 'RNN':
          g_batch = model.g(inputs).detach().cpu().numpy()
        elif options.RNN_type == 'RNN_2RNN':
          if compute_RNN_hierarchy == 'RNN1':  # decide which RNN to compute: RNN1 or RNN2
            g_batch = model.g1(inputs).detach().cpu().numpy()
          elif compute_RNN_hierarchy == 'RNN2':
            g_batch = model.g2(inputs).detach().cpu().numpy()
        elif options.RNN_type == 'RNN_reconstruction':
          g_batch = model.g((inputs[0], inputs[1], pc_outputs)).detach().cpu().numpy()
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        theta_batch = np.reshape(theta_batch.cpu(), [-1]).numpy()
        g_batch = g_batch[:,:,idxs].reshape(-1, Ng)
        
        g[index] = g_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res

        for i in range(options.batch_size*options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            distance = map2pi(bins - theta_batch[i])
            theta_idx = np.clip(np.abs(distance).argmin(), 0, len(bins) - 1)
            if x >=0 and x < res and y >=0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += g_batch[i, :]
                counts_theta[theta_idx] += 1
                activations_theta[:, theta_idx] += g_batch[i, :]
                
    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]

    for theta_idx in range(res*2):
        if counts_theta[theta_idx] > 0:
            activations_theta[:, theta_idx] /= counts_theta[theta_idx]
                
    g = g.reshape([-1, Ng])
    pos = pos.reshape([-1, 2])

    rate_map = activations.reshape(Ng, -1)
    return activations, rate_map, g, pos, activations_theta


def save_ratemaps(model, trajectory_generator, options, step, res=20, n_avg=None, compute_RNN_hierarchy=None):
    if not n_avg:
        n_avg = 1000 // options.sequence_length

    activations, rate_map, g, pos, activations_theta = compute_ratemaps(model, trajectory_generator,
                                                     options, res=res, n_avg=n_avg, compute_RNN_hierarchy=compute_RNN_hierarchy)
    rm_fig = plot_ratemaps(activations, n_plots=len(activations))
    imdir = options.save_dir + "/" + options.run_ID
    filename = imdir + "/" + str(step)
    if compute_RNN_hierarchy is not None:
        filename += "_" + str(compute_RNN_hierarchy)
    filename += ".png"
    imsave(filename, rm_fig)
    

# Calculate phase differences
def phase_difference(phases):
    phase_diff = np.angle(np.exp(1j * (phases[:, None] - phases[None, :])))
    return phase_diff

# Calculate average weights as a function of phase difference
def average_weights_vs_phase_diff(phase, J, idxs, bin_size=60):
    phases = phase[idxs]
    weights = J[idxs][:, idxs]
    phase_diff = phase_difference(phases)
    
    # Flatten the arrays
    phase_diff_flat = phase_diff.flatten()
    weights_flat = weights.flatten()
    
    # Bin the phase differences
    bins = np.linspace(-np.pi, np.pi, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    digitized = np.digitize(phase_diff_flat, bins) - 1
    
    # Calculate the average weight for each bin
    avg_weights = np.array([weights_flat[digitized == i].mean() for i in range(len(bin_centers))])
    
    return bin_centers, avg_weights

# Plot average weights as a function of phase difference
def plot_average_weights_vs_phase_diff(phase, J, ax, idxs, title, bin_size=60):
    bin_centers, avg_weights = average_weights_vs_phase_diff(phase, J, idxs, bin_size)
    ax.plot(bin_centers, avg_weights, color='black')  # Set line color to black
    ax.set_title(title)
    # ax.set_xlabel('Phase Difference')
    # ax.set_ylabel('Average Weight')
    ax.set_facecolor('white')  # Set background color to white
