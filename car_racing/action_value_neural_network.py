import numpy as np
from copy import deepcopy

class ActionValueNetwork:

    def __init__(self, network_config):

        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units_1 = network_config.get("num_hidden_units_1")
        #self.num_hidden_units_2 = network_config.get("num_hidden_units_2")
        self.num_actions = network_config.get("num_actions")
        
        self.rand_generator = np.random.RandomState(network_config.get("seed"))
        
        # NN Architecture
        self.layer_sizes = [self.state_dim, self.num_hidden_units_1, self.num_actions]
        
        # Weights
        self.weights = [dict() for i in range(0, len(self.layer_sizes) - 1)]
        for i in range(0, len(self.layer_sizes) - 1):
            self.weights[i]['W'] = self.init_saxe(self.layer_sizes[i], self.layer_sizes[i + 1])
            self.weights[i]['b'] = np.zeros((1, self.layer_sizes[i + 1]))
    
    # Compute action-value
    def get_action_values(self, s):
        """
        Args:
            s (Numpy array): The state.
        Returns:
            The action-values (Numpy array) calculated using the network's weights.
        """
        
        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0) # ReLu
        
        # W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        # psi = np.dot(x, W1) + b1
        # x = np.maximum(psi, 0) # ReLu

        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        q_vals = np.dot(x, W1) + b1

        return q_vals
    
    def get_TD_update(self, s, delta_mat):
        """
        Args:
            s (Numpy array): The state.
            delta_mat (Numpy array): A 2D array of shape (batch_size, num_actions). Each row of delta_mat  
            correspond to one state in the batch. Each row has only one non-zero element 
            which is the TD-error corresponding to the action taken.
        Returns:
            The TD update (Array of dictionaries with gradient times TD errors) for the network's weights
        """

        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        #W2, b2 = self.weights[2]['W'], self.weights[2]['b'] # New

        td_update = [dict() for i in range(len(self.weights))]
        
        psi_0 = np.dot(s, W0) + b0
        x_0 = np.maximum(psi_0, 0)
        dx_0 = (psi_0 > 0).astype(float)

        # psi_1 = np.dot(x_0, W1) + b1 # New
        # x_1 = np.maximum(psi_1, 0) # New
        # dx_1 = (psi_1 > 0).astype(float) # New
     
        v = delta_mat
        td_update[1]['W'] = np.dot(x_0.T, v) * 1. / s.shape[0]
        td_update[1]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]

        # v = np.dot(v, W2.T) * dx_1
        # td_update[1]['W'] = np.dot(x_0.T, v) * 1. / s.shape[0]
        # td_update[1]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]
        
        v = np.dot(v, W1.T) * dx_0
        td_update[0]['W'] = np.dot(s.T, v) * 1. / s.shape[0]
        td_update[0]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]
                
        return td_update
    
    def init_saxe(self, rows, cols):
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            NumPy Array consisting of weights for the layer based on the initialization in Saxe et al.
        """
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T
        return tensor
    
    def get_weights(self):
        """
        Returns: 
            A copy of the current weights of this network.
        """
        return deepcopy(self.weights)
    
    def set_weights(self, weights):
        """
        Args: 
            weights (list of dictionaries): Consists of weights that this network will set as its own weights.
        """
        self.weights = deepcopy(weights)
