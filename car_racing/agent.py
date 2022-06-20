from __future__ import print_function
from abc import ABCMeta, abstractmethod
from action_value_neural_network import ActionValueNetwork
from adam import Adam
from replay_buffer import ReplayBuffer
import numpy as np
from copy import deepcopy


"""
An abstract class that specifies the Agent API for RL-Glue-py.
"""


class BaseAgent:
    """Implements the agent for an RL-Glue environment.
    Note:
        agent_init, agent_start, agent_step, agent_end, agent_cleanup, and
        agent_message are required methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def agent_init(self, agent_info= {}):
        """Setup for the agent called when the experiment first starts."""

    @abstractmethod
    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""

    @abstractmethod
    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """

class ExpectedSarsaAgent(BaseAgent):

    def __init__(self):
        self.name = "expected_sarsa_agent"

    @staticmethod
    def softmax(action_values, tau=1.0):
        """
        Args:
            action_values (Numpy array): A 2D array of shape (batch_size, num_actions). 
                        The action-values computed by an action-value network.              
            tau (float): The temperature parameter scalar.
        Returns:
            A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
            the actions representing the policy.
        """
        
        preferences = action_values / tau
        max_preference = np.max(preferences, axis=1)

        reshaped_max_preference = max_preference.reshape((-1, 1))
        exp_preferences = np.exp(preferences - reshaped_max_preference)
        sum_of_exp_preferences = np.sum(exp_preferences,axis=1)
        
        reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
        
        # Compute the action probabilities according to the equation in the previous cell.
        action_probs = exp_preferences / reshaped_sum_of_exp_preferences
        action_probs = action_probs.squeeze()

        return action_probs

    def get_td_error(self, states, next_states, actions, rewards, discount, terminals, network, current_q, tau):
        """
        Args:
            states (Numpy array): The batch of states with the shape (batch_size, state_dim).
            next_states (Numpy array): The batch of next states with the shape (batch_size, state_dim).
            actions (Numpy array): The batch of actions with the shape (batch_size,).
            rewards (Numpy array): The batch of rewards with the shape (batch_size,).
            discount (float): The discount factor.
            terminals (Numpy array): The batch of terminals with the shape (batch_size,).
            network (ActionValueNetwork): The latest state of the network that is getting replay updates.
            current_q (ActionValueNetwork): The fixed network used for computing the targets, 
                                            and particularly, the action-values at the next-states.
        Returns:
            The TD errors (Numpy array) for actions taken, of shape (batch_size,)
        """

        # q_next_mat is a 2D array of shape (batch_size, num_actions)
        q_next_mat = current_q.get_action_values(next_states)

        # action-values in q_next_mat to softmax() (is a 2D array of shape (batch_size, num_actions))
        probs_mat = self.softmax(q_next_mat,tau)

        # Compute the estimate of the next state value, v_next_vec.
        weighted_next_mat = q_next_mat*probs_mat
        action_values_sum = np.sum(weighted_next_mat,axis=1)
        v_next_vec = action_values_sum*(1-terminals)

        # Compute Expected Sarsa target
        target_vec = rewards + discount*v_next_vec
    
        # Compute action values at the current states for all actions using network
        q_mat = network.get_action_values(states)

        # Batch Indices is an array from 0 to the batch size - 1. 
        batch_indices = np.arange(q_mat.shape[0])

        # Compute q_vec by selecting q(s, a) from q_mat for taken actions
        q_vec = q_mat[batch_indices,actions]
    
        # Compute TD errors for actions taken
        delta_vec = target_vec - q_vec
    
        return delta_vec

    def optimize_network(self, experiences, discount, optimizer, network, current_q, tau):
        """
        Args:
            experiences (Numpy array): The batch of experiences including the states, actions, 
                                    rewards, terminals, and next_states.
            discount (float): The discount factor.
            network (ActionValueNetwork): The latest state of the network that is getting replay updates.
            current_q (ActionValueNetwork): The fixed network used for computing the targets, 
                                            and particularly, the action-values at the next-states.
        """
        
        # Get states, action, rewards, terminals, and next_states from experiences
        states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
        batch_size = states.shape[0]

        # Compute TD error using the get_td_error function
        # Note that q_vec is a 1D array of shape (batch_size)
        delta_vec = self.get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau)

        # Batch Indices is an array from 0 to the batch_size - 1. 
        batch_indices = np.arange(batch_size)

        # Make a td error matrix of shape (batch_size, num_actions)
        # delta_mat has non-zero value only for actions taken
        delta_mat = np.zeros((batch_size, network.num_actions))
        delta_mat[batch_indices, actions] = delta_vec

        # Pass delta_mat to compute the TD errors times the gradients of the network's weights from back-propagation
        td_update = network.get_TD_update(states,delta_mat)

        # Pass network.get_weights and the td_update to the optimizer to get updated weights
        weights = optimizer.update_weights(network.get_weights(), td_update)

        network.set_weights(weights)
        
    def agent_init(self, agent_config):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the agent.

        Assume agent_config dict contains:
        {
            network_config: dictionary,
            optimizer_config: dictionary,
            replay_buffer_size: integer,
            minibatch_sz: integer, 
            num_replay_updates_per_step: float
            discount_factor: float,
        }
        """
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'], 
                                          agent_config['minibatch_sz'], agent_config.get("seed"))
        self.network = ActionValueNetwork(agent_config['network_config'])
        self.optimizer = Adam(self.network.layer_sizes, agent_config["optimizer_config"])
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']
        
        self.rand_generator = np.random.RandomState(agent_config.get("seed"))
        
        self.last_state = None
        self.last_action = None
        
        self.sum_rewards = 0
        self.episode_steps = 0

    def policy(self, state):
        """
        Args:
            state (Numpy array): the state.
        Returns:
            the action. 
        """
        action_values = self.network.get_action_values(state)
        probs_batch = self.softmax(action_values, self.tau)
        action = self.rand_generator.choice(self.num_actions, p=probs_batch.squeeze())
        return action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)
        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        
        self.sum_rewards += reward
        self.episode_steps += 1
        state = np.array([state])

        # Select action
        action = self.policy(state)

        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                
                # Call optimize_network to update the weights of the network
                self.optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau)
                
        # Update the last state and last action.
        self.last_state = state
        self.last_action = action
        
        return action


    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        
        # Set terminal state to an array of zeros
        state = np.zeros_like(self.last_state)

        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state,self.last_action,reward,1,state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                
                # Call optimize_network to update the weights of the network
                self.optimize_network(experiences,self.discount,self.optimizer,self.network,current_q,self.tau)     
        
    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")
