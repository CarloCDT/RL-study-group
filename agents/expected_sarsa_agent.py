import numpy as np
import pandas as pd

class ExpectedSarsaAgent():

    def __init__(self, agent_init_info):
        """
        Setup for the Expected Sarsa agent.
        
        Args:
            agent_init_info (dict), the parameters used to initialize the agent. 
            The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            alpha (float): The step-size,
            gamma (float): The discount factor,
        }
        
        """

        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.alpha = agent_init_info["alpha"]
        self.gamma = agent_init_info["gamma"]

        # Set Random Seed
        self.rand_generator = np.random.RandomState(0)
        
        # Create an array for action-value estimates and initialize it to zero.
        self.q = np.zeros((self.num_states, self.num_actions))

    def agent_start(self, observation):
        """
        The first method called when the episode starts, called after
        the environment starts.
        Args:
            observation (int): the state observation from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        
        # Choose action using epsilon greedy.
        state = observation
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (int): the state observation from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        
        # Choose action using epsilon greedy.
        state = observation
        current_q = self.q[state,:]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        q_max = np.max(self.q[state,:])
        pi = np.ones(self.num_actions) * self.epsilon / self.num_actions \
           + (self.q[state,:] == q_max) * (1 - self.epsilon) / np.sum(self.q[state,:] == q_max)
        
        expected_q = np.sum(self.q[state,:] * pi)
            
        # Update values    
        self.q[self.prev_state, self.prev_action] = self.q[self.prev_state, self.prev_action] + self.alpha \
                                                    * (reward + self.gamma * expected_q - self.q[self.prev_state, self.prev_action])
        self.prev_state = state
        self.prev_action = action
        
        return action
    
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # Update values
        self.q[self.prev_state, self.prev_action] += self.alpha * (reward- self.q[self.prev_state, self.prev_action])

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)


    def create_summary_table(self):
        """
        Create a pandas summary table based on agents q
        """
        df_summary = pd.DataFrame(self.q)
        df_summary["optimal_action"] = self.q.argmax(axis=1)
        df_summary["state_value"] = self.q.max(axis=1)
        df_summary = df_summary.reset_index().rename(columns={"index":"state"})
        
        return df_summary
        