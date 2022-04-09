import numpy as np
import pandas as pd

class DynaQAgent():

    def __init__(self, agent_info):
        """
        Setup for the agent called when the experiment first starts.

        Args:
            agent_info (dict), the parameters used to initialize the agent. The dictionary contains:
            {
                num_states (int): The number of states,
                num_actions (int): The number of actions,
                epsilon (float): The parameter for epsilon-greedy exploration,
                alpha (float): The step-size,
                gamma (float): The discount factor,
                planning_steps (int): The number of planning steps per environmental interaction

                random_seed (int): the seed for the RNG used in epsilon-greedy
                planning_random_seed (int): the seed for the RNG used in the planner
            }
        """
        # Store the parameters provided in agent_info.
        self.num_states = agent_info["num_states"]
        self.num_actions = agent_info["num_actions"]
        self.epsilon = agent_info["epsilon"]
        self.alpha = agent_info["alpha"]
        self.gamma = agent_info["gamma"]
        self.planning_steps = agent_info["planning_steps"]

        # Set Random Seed
        self.rand_generator = np.random#.RandomState(0)
        self.planning_rand_generator = np.random#.RandomState(11)


        # Create an array for action-value and actions.
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1

         # model is a dictionary of dictionaries, which maps states to actions to (reward, next_state) tuples
        self.model = {}

    def update_model(self, past_state, past_action, state, reward):
        """
        Updates the model.
        
        Args:
            past_state       (int): s
            past_action      (int): a
            state            (int): s'
            reward           (int): r
        Returns:
            Nothing
        """
        self.model[past_state] = self.model.get(past_state, {}) 
        self.model[past_state][past_action] = state, reward

    def planning_step(self):
        """
        Performs planning, i.e. indirect RL.

        Args:
            None
        Returns:
            Nothing
        """
        
        for i in range(self.planning_steps):
            
            past_state = self.planning_rand_generator.choice(list(self.model.keys())) # Choose state
            past_action = self.planning_rand_generator.choice(list(self.model[past_state].keys())) # Choose action
            
            next_state, reward = self.model[past_state][past_action] # predicted next state and reward
        
            if next_state == -1:
                q_max = 0
            else:
                q_max = np.max(self.q_values[next_state])
        
            self.q_values[past_state, past_action] += self.alpha * (reward + self.gamma * q_max - self.q_values[past_state, past_action])

    def argmax(self, q_values):
        """
        argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action values
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

    def choose_action_egreedy(self, state):
        """
        Returns an action using an epsilon-greedy policy w.r.t. the current action-value function.

        Important: assume you have a random number generator 'rand_generator' as a part of the class
                    which you can use as self.rand_generator.choice() or self.rand_generator.rand()

        Args:
            state (List): coordinates of the agent (two elements)
        Returns:
            The action taken w.r.t. the aforementioned epsilon-greedy policy
        """

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.actions)
        else:
            values = self.q_values[state]
            action = self.argmax(values)

        return action

    def agent_start(self, state):
        """
        The first method called when the experiment starts, 
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        
        action = self.choose_action_egreedy(state) # Choose action in state
        
        self.past_state = state
        self.past_action = action

        return self.past_action

    def agent_step(self, reward, state):
        """
        A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """
        
        q_max = np.max(self.q_values[state])
        self.q_values[self.past_state, self.past_action] += self.alpha * (reward + self.gamma * q_max - self.q_values[self.past_state, self.past_action])
        
        # Model update step
        self.update_model(self.past_state, self.past_action, state, reward)
        
        # Planning
        self.planning_step()
        
        # Choose Action
        action = self.choose_action_egreedy(state)
        
        # Save the current state and action
        self.past_state = state
        self.past_action = action
        
        return self.past_action

    def agent_end(self, reward):
        """
        Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
           
        # Direct-RL step
        self.q_values[self.past_state, self.past_action] += self.alpha * (reward - self.q_values[self.past_state, self.past_action])
        
        # Model Update step
        self.update_model(self.past_state, self.past_action, -1, reward)
        
        # Planning
        self.planning_step()

    def create_summary_table(self):
        """
        Create a pandas summary table based on agents q
        """
        df_summary = pd.DataFrame(self.q_values)
        df_summary["optimal_action"] = self.q_values.argmax(axis=1)
        df_summary["state_value"] = self.q_values.max(axis=1)
        df_summary = df_summary.reset_index().rename(columns={"index":"state"})
        
        return df_summary
