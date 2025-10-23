import numpy as np
import torch.cuda
from utils import generate_run_ID
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN, RNN_2RNN, RNN_reconstruction
from trainer import Trainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    default='models/',
                    help='directory to save trained models')
parser.add_argument('--n_epochs',
                    type=int, default=100,
                    help='number of training epochs')
parser.add_argument('--n_steps',
                    type=int, default=1000,
                    help='batches per epoch')
parser.add_argument('--batch_size',
                    type=int,default=200,
                    help='number of trajectories per batch')
parser.add_argument('--sequence_length',
                    type=int, default=20,
                    help='number of steps in trajectory')
parser.add_argument('--learning_rate',
                    type=float, default=1e-4,
                    help='gradient descent learning rate')
parser.add_argument('--Np',
                    type=int, default=512,
                    help='number of place cells')
parser.add_argument('--Ng',
                    type=int, default=4096,
                    help='number of grid cells')
parser.add_argument('--place_cell_rf',
                    type=float, default=0.12,
                    help='width of place cell center tuning curve (m)')
parser.add_argument('--surround_scale',
                    type=float, default=2,
                    help='if DoG, ratio of sigma2^2 to sigma1^2')
parser.add_argument('--RNN_type',
                    default='RNN',
                    help='RNN, RNN_2RNN or RNN_reconstruction')
parser.add_argument('--activation',
                    default='relu',
                    help='recurrent nonlinearity')
parser.add_argument('--weight_decay',
                    default=1e-4,
                    help='strength of weight decay on recurrent weights')
parser.add_argument('--DoG', 
                    default=True,
                    help='use difference of gaussians tuning curves')
parser.add_argument('--periodic',
                    default=False,
                    help='trajectories with periodic boundary conditions')
parser.add_argument('--box_width',
                    type=float, default=2.2,
                    help='width of training environment')
parser.add_argument('--box_height',
                    type=float, default=2.2,
                    help='height of training environment')
parser.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='device to use for training')
parser.add_argument('--seed',
                    default=None,
                    help='random seed; if None, use example pc centers')

options = parser.parse_args()
options.run_ID = generate_run_ID(options)

print(f'Using device: {options.device}')

place_cells = PlaceCells(options)

if options.RNN_type == 'RNN':
    model = RNN(options, place_cells)
elif options.RNN_type == 'RNN_2RNN':
    model = RNN_2RNN(options, place_cells)
elif options.RNN_type == 'RNN_reconstruction':
    model = RNN_reconstruction(options, place_cells)

# Put model on GPU if using GPU
model = model.to(options.device)

trajectory_generator = TrajectoryGenerator(options, place_cells)

trainer = Trainer(options, model, trajectory_generator)

# Train
trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps)
