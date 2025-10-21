# -*- coding: utf-8 -*-
import torch
import numpy as np

class Pruner(object):
  def __init__(self, model, trajectory_generator):
    self.model = model
    self.trajectory_generator = trajectory_generator

    self.loss = []
    self.err = []

  # Pruning velocity inputs
  def prune_step_velocity(self, inputs, pos, mask):
    """
    Perform pruning on one batch of trajectories for velocity inputs.

    Args:
      inputs: Batch of 2D velocity inputs with shape [batch_size, sequence_length, 2].
      pos: Ground truth 2D positions with shape [batch_size, sequence_length, 2].
      mask: Pruning mask for velocity inputs.

    Returns:
      loss: Average loss for this batch.
      err: Average decoded position error in cm.
    """

    self.model.zero_grad()
    err = self.model.compute_err_prune_velocity(inputs, pos, mask)

    return err.item()

  def prune_velocity(self, mask, n_steps=10, mask_size=100, interval=10):
    """
    Perform pruning on velocity inputs over multiple steps.

    Args:
      mask: Initial pruning mask.
      n_steps: Number of training steps per pruning level.
      mask_size: Total number of elements to prune.
      interval: Interval for pruning levels.

    Returns:
      errs: List of errors for each pruning level and step.
    """

    errs = np.zeros((1 + mask_size // interval, n_steps))  # Create a 2D array to store errors

    for i in range(0, errs.shape[0]):  # Take every `interval` steps
      prune_mask = np.ones(self.model.Ng).astype("float32")  # Start with no pruning
      prune_inds = np.random.choice(mask, size=i*interval, replace=False)
      prune_mask[prune_inds] = 0

      # Construct generator
      gen = self.trajectory_generator.get_generator()

      # Iterate n_steps times for each pruning level
      for j in range(n_steps):
        inputs, _, pos = next(gen)
        err = self.prune_step_velocity(inputs, pos, prune_mask)
        errs[i, j] = err  # Adjust indexing to match the reduced loop range

    self.err = errs.tolist()  # Convert to list if needed for further processing

    return self.err
  

  # Pruning read-out connections
  def prune_step_read_out(self, inputs, pos, mask):
    """
    Perform pruning on one batch of trajectories for read-out connections.

    Args:
      inputs: Batch of 2D velocity inputs with shape [batch_size, sequence_length, 2].
      pos: Ground truth 2D positions with shape [batch_size, sequence_length, 2].
      mask: Pruning mask for read-out connections.

    Returns:
      err: Average decoded position error in cm.
    """

    self.model.zero_grad()
    err = self.model.compute_err_prune_read_out(inputs, pos, mask)

    return err

  def prune_read_out(self, mask, n_steps=10, mask_size=100, step_by_step=True, interval=10):
    """
    Perform pruning on read-out connections over multiple steps.

    Args:
      mask: Initial pruning mask.
      n_steps: Number of training steps per pruning level.
      mask_size: Total number of elements to prune.
      step_by_step: Whether to prune incrementally or all at once.
      interval: Interval for pruning levels.

    Returns:
      errs: List of errors for each pruning level and step.
    """

    if step_by_step:
      errs = np.zeros((1 + mask_size // interval, n_steps))
      for i in range(0, errs.shape[0]):  # Take every `interval` steps
        prune_inds = np.random.choice(mask, size=i*interval, replace=False)

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # Iterate n_steps times for each pruning level
        for j in range(n_steps):
          inputs, _, pos = next(gen)
          err = self.prune_step_read_out(inputs, pos, prune_inds)
          errs[i, j] = err

      self.err = errs.tolist()

    else:
      errs = []
      prune_inds = np.random.choice(mask, size=mask_size, replace=False)

      # Construct generator
      gen = self.trajectory_generator.get_generator()

      # Iterate n_steps times for pruning
      for j in range(n_steps):
        inputs, _, pos = next(gen)
        err = self.prune_step_read_out(inputs, pos, prune_inds)
        errs.append(err)

      errs = torch.stack(errs)  # Convert list to tensor
      self.err = errs.cpu().numpy()

    return self.err


  # Pruning recurrent connections
  def prune_step_recurrent(self, inputs, pos, mask_1, mask_2):
    """
    Perform pruning on one batch of trajectories for recurrent connections.

    Args:
      inputs: Batch of 2D velocity inputs with shape [batch_size, sequence_length, 2].
      pos: Ground truth 2D positions with shape [batch_size, sequence_length, 2].
      mask_1: Pruning mask for the first set of recurrent connections.
      mask_2: Pruning mask for the second set of recurrent connections.

    Returns:
      err: Average decoded position error in cm.
    """

    self.model.zero_grad()
    err = self.model.compute_err_prune_recurrent(
      inputs, pos, mask_1, mask_2
    )

    return err.item()

  def prun_recurrent(self, mask_1, mask_2, n_steps=10, mask_size=100, step_by_step=True, interval=10):
    """
    Perform pruning on recurrent connections over multiple steps.

    Args:
      mask_1: Initial pruning mask for the first set of recurrent connections.
      mask_2: Initial pruning mask for the second set of recurrent connections.
      n_steps: Number of training steps per pruning level.
      mask_size: Total number of elements to prune.
      step_by_step: Whether to prune incrementally or all at once.
      interval: Interval for pruning levels.

    Returns:
      errs: List of errors for each pruning level and step.
    """

    if step_by_step:
      errs = np.zeros((1 + mask_size // interval, n_steps))
      for i in range(0, errs.shape[0]):  # Take every `interval` steps
        prune_inds_1 = mask_1.copy()
        prune_inds_2 = np.random.choice(mask_2, size=i*interval, replace=False)

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # Iterate n_steps times for each pruning level
        for j in range(n_steps):
          inputs, _, pos = next(gen)
          err = self.prune_step_recurrent(
            inputs, pos, prune_inds_1, prune_inds_2
          )
          errs[i, j] = err

      self.err = errs.tolist()

    else:
      errs = np.zeros((n_steps))
      prune_inds_1 = np.random.choice(mask_1, size=len(mask_1), replace=False)
      prune_inds_2 = np.random.choice(mask_2, size=len(mask_2), replace=False)

      # Construct generator
      gen = self.trajectory_generator.get_generator()

      # Iterate n_steps times for pruning
      for j in range(n_steps):
        inputs, _, pos = next(gen)
        err = self.prune_step_recurrent(
          inputs, pos, prune_inds_1, prune_inds_2
        )
        errs[j] = err

      self.err = errs.tolist()

    return self.err


  # Pruning each cell type
  def prune_step_ablate(self, inputs, pos, mask):
    """
    Perform pruning on one batch of trajectories for specific cell types.

    Args:
      inputs: Batch of 2D velocity inputs with shape [batch_size, sequence_length, 2].
      pos: Ground truth 2D positions with shape [batch_size, sequence_length, 2].
      mask: Pruning mask for specific cell types.

    Returns:
      err: Average decoded position error in cm.
    """

    self.model.zero_grad()
    err = self.model.compute_err_prune_ablate(inputs, pos, mask)
    
    return err.item()

  def prune_ablate(self, mask, n_steps=10, mask_size=100, step_by_step=True, interval=10):
    """
    Perform pruning on specific cell types over multiple steps.

    Args:
      mask: Initial pruning mask.
      n_steps: Number of training steps per pruning level.
      mask_size: Total number of elements to prune.
      step_by_step: Whether to prune incrementally or all at once.
      interval: Interval for pruning levels.

    Returns:
      errs: List of errors for each pruning level and step.
    """
    
    if step_by_step:
      errs = np.zeros((1 + mask_size // interval, n_steps))
      for i in range(0, errs.shape[0]):  # Take every `interval` steps
        prune_inds = np.random.choice(mask, size=i*interval, replace=False)
        
        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # Iterate n_steps times for each pruning level
        for j in range(n_steps):
          inputs, _, pos = next(gen)
          err = self.prune_step_ablate(inputs, pos, prune_inds)
          errs[i, j] = err

      self.err = errs.tolist() 
    
    else:
      errs = np.zeros((n_steps)) 
      prune_inds = np.random.choice(mask, size=mask_size, replace=False)

      # Construct generator
      gen = self.trajectory_generator.get_generator()

      # Iterate n_steps times for pruning
      for j in range(n_steps):
        inputs, _, pos = next(gen)
        err = self.prune_step_ablate(inputs, pos, prune_inds)
        errs[j] = err

      self.err = errs.tolist() 
      
    return self.err
