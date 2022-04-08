import numpy as np
import pandas as pd

class MonteCarloExploringStartsAgent():

    def __init__(self, agent_init_info):
        """
        Setup for the Monte Carlo exploring strats agent.
        
        Args:
            agent_init_info (dict), the parameters used to initialize the agent. 
            The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            gamma (float): The discount factor,
        }
        
        """

        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.gamma = agent_init_info["gamma"]

        # Initialize 
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        # Set Random Seed
        self.rand_generator = np.random.RandomState(0)
        
        # Create an array for state-action value estimates and initialize it to zero.
        self.q = np.zeros((self.num_states, self.num_actions))
        
        # Create returns array
        self.returns = {}
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.returns[(s,a)] = [] 
   
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
        
        # Choose random action
        state = observation
        action = self.rand_generator.randint(self.num_actions)

        # Update values
        self.episode_states.append(state)
        self.episode_actions.append(action)

        return action
    
    def agent_step(self, reward, observation):
        """
        A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (int): the state observation from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        
        # Choose action using greedy.
        state = observation
        current_q = self.q[state, :]
        action = self.argmax(current_q)
                
        # Update values
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

        return action
    
    def agent_end(self, reward):
        """
        Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Append last reward
        self.episode_rewards.append(reward)

        # Perform the last update in the episode
        G = 0

        for idx in range(len(self.episode_states)):
            G = self.gamma*G + self.episode_rewards[-(idx+1)]

            S_t = self.episode_states[-(idx+1)]
            A_t = self.episode_actions[-(idx+1)]

            if (S_t, A_t) not in zip(self.episode_states[:-(idx+1)], self.episode_actions[:-(idx+1)]):
                self.returns[(S_t, A_t)].append(G)
                self.q[S_t, A_t] = np.mean(self.returns[(S_t, A_t)])

        # Delete history
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def argmax(self, q_values):
        """
        argmax with random tie-breaking
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
        Create a pandas sumamry table based on agents q
        """
        df_summary = pd.DataFrame(self.q)
        df_summary["optimal_action"] = self.q.argmax(axis=1)
        df_summary["state_value"] = self.q.max(axis=1)
        df_summary = df_summary.reset_index().rename(columns={"index":"state"})
        
        return df_summary
