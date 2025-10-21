# -*- coding: utf-8 -*-
import torch

# Vanilla RNN model for path integration task.
class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells
        self.activation = options.activation
        self.periodic = options.periodic

        # Random seed for reproducibility
        if options.seed is not None:
          torch.manual_seed(options.seed)

        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.RNN = torch.nn.RNN(input_size=2,
                                hidden_size=self.Ng,
                                nonlinearity=self.activation,
                                bias=False)
        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        
        self.softmax = torch.nn.Softmax(dim=-1)


    def g(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g,_ = self.RNN(v, init_state)
        return g
    

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [batch_size, sequence_length, Np].
        '''
        place_preds = self.decoder(self.g(inputs))
        
        return place_preds


    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(preds)
        loss = -(y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        ## Consider periodic boundaries with environment size
        if self.periodic:
            diff = torch.remainder(pos - pred_pos + self.box_width/2, self.box_width) - self.box_width/2
        else:
            diff = pos - pred_pos
        ## Compute the error
        err = torch.sqrt((diff**2).sum(-1)).mean()

        return loss, err
    

    def set_weights(self, weights):
      '''
      Set the weights of the model.
      Args:
        weights: List of weight matrices [encoder_weights, RNN_input_weights, 
             RNN_hidden_weights, decoder_weights].
      '''
      self.encoder.weight.data = torch.tensor(weights[0].T)
      self.RNN.weight_ih_l0.data = torch.tensor(weights[1].T)
      self.RNN.weight_hh_l0.data = torch.tensor(weights[2].T)
      self.decoder.weight.data = torch.tensor(weights[3].T)

    # Prune velocity input connections
    def prune_forward_v(self, inputs, mask):
      '''
      Compute grid cell activations with pruned velocity inputs.
      Args:
        inputs: Batch of 2D velocity inputs with shape [batch_size, sequence_length, 2].
        mask: Mask to prune velocity input connections.

      Returns: 
        g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
      '''
      with torch.no_grad():
        v, p0 = inputs
        g = [self.encoder(p0)[None]]  # Initial state

        # Get RNN weights
        W_ih = self.RNN.weight_ih_l0  # Input-to-hidden weights
        W_hh = self.RNN.weight_hh_l0  # Hidden-to-hidden weights
        
        # Manually implement the RNN forward pass
        for i in range(v.shape[0]):  # v.shape[1] is the sequence length
          # Compute the current time step hidden state
          velocity_input = torch.matmul(v[i], W_ih.t()) * mask
          h_t = torch.relu(velocity_input + torch.matmul(g[-1], W_hh.t()))
          g.append(h_t)
      return torch.stack(g[1:], dim=1)[0]

    # Compute error with pruned velocity inputs
    def compute_err_prune_velocity(self, inputs, pos, mask):
      '''
      Compute average decoding error with pruned velocity inputs.
      Args:
        inputs: Batch of 2D velocity inputs with shape [batch_size, sequence_length, 2].
        pc_outputs: Ground truth place cell activations with shape 
              [batch_size, sequence_length, Np].
        pos: Ground truth 2D position with shape [batch_size, sequence_length, 2].
        mask: Mask to prune velocity input connections.

      Returns:
        err: Average decoded position error in cm.
      '''
      preds = self.decoder(self.prune_forward_v(inputs, mask))

      # Compute decoding error
      pred_pos = self.place_cells.get_nearest_cell_pos(preds)
      err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

      return err

    # Compute error with pruned read-out connections
    def compute_err_prune_read_out(self, inputs, pos, mask):
      '''
      Compute average decoding error with pruned read-out connections.
      Args:
        inputs: Batch of 2D velocity inputs with shape [batch_size, sequence_length, 2].
        pc_outputs: Ground truth place cell activations with shape 
              [batch_size, sequence_length, Np].
        pos: Ground truth 2D position with shape [batch_size, sequence_length, 2].
        mask: Mask to prune read-out connections.

      Returns:
        err: Average decoded position error in cm.
      '''
      original_weights = self.decoder.weight.data.clone()

      with torch.no_grad():
        self.decoder.weight.data[:, mask] = 0

      preds = self.predict(inputs)

      # Compute decoding error
      pred_pos = self.place_cells.get_nearest_cell_pos(preds)
      err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()
      
      # Restore original weights
      self.decoder.weight.data = original_weights

      return err

    # Compute error with pruned recurrent connections
    def compute_err_prune_recurrent(self, inputs, pos, mask_1, mask_2):
      '''
      Compute average decoding error with pruned recurrent connections.
      Args:
        inputs: Batch of 2D velocity inputs with shape [batch_size, sequence_length, 2].
        pc_outputs: Ground truth place cell activations with shape 
              [batch_size, sequence_length, Np].
        pos: Ground truth 2D position with shape [batch_size, sequence_length, 2].
        mask_1: Mask for source neurons in recurrent connections.
        mask_2: Mask for target neurons in recurrent connections.

      Returns:
        err: Average decoded position error in cm.
      '''
      original_weights = self.RNN.weight_hh_l0.data.clone()

      with torch.no_grad():
        mask_1 = torch.tensor(mask_1) if not isinstance(mask_1, torch.Tensor) else mask_1
        mask_2 = torch.tensor(mask_2) if not isinstance(mask_2, torch.Tensor) else mask_2
        mask = torch.cartesian_prod(mask_2, mask_1)  # Generate all combinations of indices
        self.RNN.weight_hh_l0.data[mask[:, 0], mask[:, 1]] = 0  # Set connections from mask_1 to mask_2 to 0
        
      preds = self.predict(inputs)

      # Compute decoding error
      pred_pos = self.place_cells.get_nearest_cell_pos(preds)
      err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()
      
      # Restore original weights
      self.RNN.weight_hh_l0.data = original_weights

      return err

    # Compute error with ablated cell types
    def compute_err_prune_ablate(self, inputs, pos, mask):
        '''
        Compute average decoding error with ablated cell types.
        Args:
          inputs: Batch of 2D velocity inputs with shape [batch_size, sequence_length, 2].
          pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
          pos: Ground truth 2D position with shape [batch_size, sequence_length, 2].
          mask: Mask to ablate specific cell types.

        Returns:
          err: Average decoded position error in cm.
        '''
        original_input = self.RNN.weight_ih_l0.data.clone()
        original_RNN = self.RNN.weight_hh_l0.data.clone()
        original_output = self.decoder.weight.data.clone()

        with torch.no_grad():

            # Perform pruning
            self.RNN.weight_ih_l0.data[mask, :] = 0
            self.RNN.weight_hh_l0.data[:, mask] = 0
            self.RNN.weight_hh_l0.data[mask, :] = 0
            self.decoder.weight.data[:, mask] = 0

        preds = self.predict(inputs)

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        # recover the original weights
        self.RNN.weight_ih_l0.data = original_input
        self.RNN.weight_hh_l0.data = original_RNN
        self.decoder.weight.data = original_output

        return err


# Figure 3b: A new RNN model taking the hierarchical pathway as a prior.
class RNN_2RNN(RNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RNN1 now takes both velocity input and feedback from RNN2
        self.RNN1 = torch.nn.RNN(input_size=2 + self.Ng,
                                hidden_size=self.Ng,
                                nonlinearity=self.activation,
                                bias=False)
        
        self.RNN2 = torch.nn.RNN(input_size=self.Ng,
                        hidden_size=self.Ng,
                        nonlinearity=self.activation,
                        bias=False)
        
    def g1(self, inputs, feedback=None):
        '''
        Compute grid cell activations with feedback from RNN2.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            feedback: Feedback from RNN2 with shape [batch_size, sequence_length, Ng].

        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        v, p0 = inputs
        sequence_length = v.size(0)
        batch_size = v.size(1)
        
        # Prepare initial state
        init_state = self.encoder(p0)[None]  # [1, batch_size, Ng]
        
        # Prepare input with feedback
        if feedback is None:
            # For first step, use zeros as feedback
            feedback = torch.zeros(sequence_length, batch_size, self.Ng, device=v.device)
        
        # Concatenate velocity and feedback
        rnn1_input = torch.cat([v, feedback], dim=-1)
        
        g, _ = self.RNN1(rnn1_input, init_state)
        return g
    
    def g2(self, inputs):
        _, p0 = inputs
        init_state = self.encoder(p0)[None]
        g, _ = self.RNN2(self.g1(inputs), init_state)
        return g

    def predict(self, inputs):
        '''
        Predict place cell code with feedback loop between RNN1 and RNN2.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [batch_size, sequence_length, Np].
        '''
        # Initialize feedback
        sequence_length = inputs[0].size(0)
        batch_size = inputs[0].size(1)
        feedback = torch.zeros(sequence_length, batch_size, self.Ng, device=inputs[0].device)
        
        # Iterate through time steps to allow feedback
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        
        # Initialize hidden states
        h1 = init_state
        h2 = init_state
        
        outputs = []
        for t in range(sequence_length):
            # Prepare RNN1 input: current velocity + feedback from previous step
            rnn1_input = torch.cat([v[t:t+1, :, :], feedback[t:t+1, :, :]], dim=-1)
            
            # RNN1 step
            out1, h1 = self.RNN1(rnn1_input, h1)
            
            # RNN2 step
            out2, h2 = self.RNN2(out1, h2)
            
            # Update feedback for next step
            if t < sequence_length - 1:
                feedback[t+1, :, :] = out2.squeeze(0)
            
            # Store output
            outputs.append(out2)
        
        # Stack all outputs
        g2_output = torch.cat(outputs, dim=0)
        place_preds = self.decoder(g2_output)
        
        return place_preds

    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(preds)
        loss = -(y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN1.weight_hh_l0**2 + self.RNN2.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        ## Consider periodic boundaries with environment size
        if self.periodic:
            diff = torch.remainder(pos - pred_pos + self.box_width/2, self.box_width) - self.box_width/2
        else:
            diff = pos - pred_pos
        ## Compute the error
        err = torch.sqrt((diff**2).sum(-1)).mean()

        return loss, err


# Figure 3d: An RNN model for a location reconstruction task.
class RNN_reconstruction(RNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Modify RNN to take place cell inputs
        self.RNN = torch.nn.RNN(input_size=self.Ng,
                        hidden_size=self.Ng,
                        nonlinearity=self.activation,
                        bias=False)

    def g(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
                
        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        _, p0, pc_outputs = inputs
        init_state = self.encoder(p0)[None]
        p_state = self.encoder(pc_outputs)

        g,_ = self.RNN(p_state, init_state)
        return g

    
    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        inputs = (inputs[0], inputs[1], pc_outputs)
        preds = self.predict(inputs)
        yhat = self.softmax(preds)
        loss = -(y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        ## Consider periodic boundaries with environment size
        if self.periodic:
            diff = torch.remainder(pos - pred_pos + self.box_width/2, self.box_width) - self.box_width/2
        else:
            diff = pos - pred_pos
        ## Compute the error
        err = torch.sqrt((diff**2).sum(-1)).mean()

        return loss, err