#!/usr/bin/env python

"""Glues together an experiment, agent, and environment.
"""

from __future__ import print_function
import numpy as np

class RLGlue:
    """RLGlue class

    args:
        env_name (string): the name of the module where the Environment class can be found
        agent_name (string): the name of the module where the Agent class can be found
    """

    def __init__(self, env_class, agent_class):
        self.environment = env_class()
        self.agent = agent_class()

        self.total_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None

        # Set front of Car
        self.car_front = [65, 48]

    @staticmethod
    def is_road(pixel):
        if pixel[1]>pixel[0]*1.3 and pixel[1]>pixel[2]*1.3:
            # Not grey (probably?)
            return False
        else:
            return True

    @staticmethod
    def is_void(pixel):
        if max(pixel)<10:
            # Not grey (probably?)
            return True
        else:
            return False

    def calculate_speed(self, input_img):
        #return self.environment.get_speed()
        return self.rl_env_message('get_speed')

    def transform_state(self, last_state):

        if len(last_state)!=8: # I should be able to do this in a better way!
            
            # Override state

            # Speed
            speed = self.calculate_speed(last_state)

            # Front Sensor
            front_sensor = self.car_front[0]
            for i in range(self.car_front[0]):
                pixel = last_state[self.car_front[0]-i, self.car_front[1], :]
                if not self.is_road(pixel):
                    front_sensor = i
                    break

            # Front left sensor
            left_sensor = self.car_front[0]
            for i in range(self.car_front[0]):
                pixel = last_state[self.car_front[0]-i, self.car_front[1]-i//4, :]
                if not self.is_road(pixel):
                    left_sensor = i
                    break
                
            # Full left sensor
            full_left_sensor = self.car_front[1]
            for i in range(self.car_front[1]):
                pixel = last_state[self.car_front[0], self.car_front[1]-i, :]
                if not self.is_road(pixel):
                    full_left_sensor =  i
                    break

            # Front Right Sensor
            right_sensor = self.car_front[0]
            for i in range(self.car_front[0]):
                pixel = last_state[self.car_front[0]-i, self.car_front[1]+i//4, :]
                if not self.is_road(pixel):
                    right_sensor = i
                    break
            
            # Full right sensor
            full_right_sensor = 96-self.car_front[1]
            for i in range(96-self.car_front[1]):
                pixel = last_state[self.car_front[0], self.car_front[1]+i, :]
                if not self.is_road(pixel):
                    full_right_sensor = i
                    break

            ## Extra Sensors
            front_road_sensor = self.car_front[0]
            for i in range(self.car_front[0]):
                pixel = last_state[self.car_front[0]-i, self.car_front[1], :]
                if self.is_road(pixel) or np.sum(pixel)==0:
                    front_road_sensor = i
                    break

            car_back = [self.car_front[0]+12, self.car_front[1]]
            back_road_sensor = 96-car_back[0]
            for i in range(96-car_back[0]):
                pixel = last_state[car_back[0]+i, car_back[1], :]
                if self.is_road(pixel) or np.sum(pixel)==0:
                    back_road_sensor = 0
                    break

            return [speed, front_sensor, left_sensor, full_left_sensor, right_sensor, full_right_sensor]
        
        else:
            print("WARNING: Calling transform_state in a wrong place!")
            return last_state

    def rl_init(self, agent_init_info={}, env_init_info={}):
        """Initial method called when RLGlue experiment is created"""
        self.environment.env_init(env_init_info)
        self.agent.agent_init(agent_init_info)

        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def rl_start(self, agent_start_info={}, env_start_info={}):
        """Starts RLGlue experiment

        Returns:
            tuple: (state, action)
        """
        
        self.total_reward = 0.0
        self.num_steps = 1

        last_state = self.environment.env_start()
        
        # New
        last_state = self.transform_state(last_state)

        self.last_action = self.agent.agent_start(last_state)

        observation = (last_state, self.last_action)

        return observation

    def rl_agent_start(self, observation):
        """Starts the agent.

        Args:
            observation: The first observation from the environment

        Returns:
            The action taken by the agent.
        """
        observation = self.transform_state(observation) # new
        return self.agent.agent_start(observation)

    def rl_agent_step(self, reward, observation):
        """Step taken by the agent

        Args:
            reward (float): the last reward the agent received for taking the
                last action.
            observation : the state observation the agent receives from the
                environment.

        Returns:
            The action taken by the agent.
        """
        observation = self.transform_state(observation) # new
        return self.agent.agent_step(reward, observation)

    def rl_agent_end(self, reward):
        """Run when the agent terminates

        Args:
            reward (float): the reward the agent received when terminating
        """
        self.agent.agent_end(reward)

    def rl_env_start(self):
        """Starts RL-Glue environment.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination
        """
        self.total_reward = 0.0
        self.num_steps = 1

        this_observation = self.environment.env_start()

        return this_observation

    def rl_env_step(self, action):
        """Step taken by the environment based on action from agent

        Args:
            action: Action taken by agent.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination.
        """
        ro = self.environment.env_step(action)
        (this_reward, _, terminal) = ro

        self.total_reward += this_reward

        if terminal:
            self.num_episodes += 1
        else:
            self.num_steps += 1

        return ro

    def rl_step(self):
        """Step taken by RLGlue, takes environment step and either step or
            end by agent.

        Returns:
            (float, state, action, Boolean): reward, last state observation,
                last action, boolean indicating termination
        """
        # acaaaaa
        (reward, last_state, term) = self.environment.env_step(self.last_action)
        
        # New
        last_state = self.transform_state(last_state)

        self.total_reward += reward;

        if term:
            self.num_episodes += 1
            self.agent.agent_end(reward)
            roat = (reward, last_state, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.agent_step(reward, last_state)
            roat = (reward, last_state, self.last_action, term)

        return roat

    def rl_cleanup(self):
        """Cleanup done at end of experiment."""
        self.environment.env_cleanup()
        self.agent.agent_cleanup()

    def rl_agent_message(self, message):
        """Message passed to communicate with agent during experiment

        Args:
            message: the message (or question) to send to the agent

        Returns:
            The message back (or answer) from the agent

        """

        return self.agent.agent_message(message)

    def rl_env_message(self, message):
        """Message passed to communicate with environment during experiment

        Args:
            message: the message (or question) to send to the environment

        Returns:
            The message back (or answer) from the environment

        """
        return self.environment.env_message(message)

    def rl_episode(self, max_steps_this_episode):
        """Runs an RLGlue episode

        Args:
            max_steps_this_episode (Int): the maximum steps for the experiment to run in an episode

        Returns:
            Boolean: if the episode should terminate
        """
        is_terminal = False

        self.rl_start()

        while (not is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.rl_step()
            is_terminal = rl_step_result[3]

        return is_terminal

    def rl_return(self):
        """The total reward

        Returns:
            float: the total reward
        """
        return self.total_reward

    def rl_num_steps(self):
        """The total number of steps taken

        Returns:
            Int: the total number of steps taken
        """
        return self.num_steps

    def rl_num_episodes(self):
        """The number of episodes

        Returns
            Int: the total number of episodes

        """
        return self.num_episodes
