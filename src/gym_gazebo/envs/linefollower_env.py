import cv2
import rclpy
import time
import gym_gazebo
import gymnasium as gym
import numpy as np
from cv_bridge import CvBridge
from gym_gazebo.core.gazebo_env import GazeboEnv
from rclpy.node import Node as RosNode
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist as RosTwist

DRIVE_SPEED = 0.2
TURN_SPEED = 0.3
DRIVE_WHILE_TURN_SPEED = 0.1

NUM_OBS_STATES = 3
NUM_ACTIONS = 3

class LineFollowerEnv(GazeboEnv):
    def __init__(self):
        # Intializing ROS2
        rclpy.init(args=None)
        
        super(LineFollowerEnv, self).__init__(launch_pkg='linefollow_ros',
                                    launch_file='my_agent.launch.xml',
                                    world_name='monza_gym')

        # Setting up ROS2 node
        self.ros_node = RosNode('line_follower_RL')

        # Initializing variables
        self.is_resetting = False
        self.latest_obs = None
        self.new_obs_event = False

        # Setting the observation and action spaces
        self.observation_space = gym.spaces.Discrete(NUM_OBS_STATES)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # ROS2 topics
        observation_topic = "/camera/image_raw"
        velocity_topic = "/cmd_vel"

        # Publishing and subscribing to ROS2 topics
        self.pub_cmd_vel_msg = self.ros_node.create_publisher(
            RosTwist, velocity_topic, 10)
            
        self.sub_obs = self.ros_node.create_subscription(
            RosImage, observation_topic, self.callback_obs_feed_, 10)


    def callback_obs_feed_(self, msg: RosImage):
        '''
        @brief observation callback resets the new_obs_event flag
        @param msg: Image from Gazebo sim
        '''
        # Takes in observation, returns a flag an observation is recieved
        if not self.is_resetting:
            self.latest_obs = msg
            self.new_obs_event = True


    ## MAIN FUNCTIONS:
    def reset(self, seed=None, options=None):
        '''
        @brief resets the simulation
        '''
        self.is_resetting = True
        print("Resetting")

        # Zero out velocities
        vel_cmd = RosTwist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Reset position and joints
        self._reset_agents()

        # Unpause the sim
        self._pause_sim(False)

        # Allow some time to settle
        time.sleep(0.1) # Wait 100 ms
        
        self.is_resetting = False

        # Wait for next observation to come in before kickstarting the data loop
        self.new_obs_event = False
        while not self.new_obs_event:
            # Telling executor (node) to spin
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # # Pause the sim after the first observation comes in
        self._pause_sim(True)
        
        state, _ = self.process_obs(self.latest_obs)

        print("Done resetting")
        return state, {}
    
    
    def step(self, action: int):
        '''
        @brief takes a step in the simulation
        @param action
        @retval state
        @retval reward
        @retval terminated
        @retval truncated
        '''
        # Process the action
        vel_cmd = self.process_action(action)

        # Unpause the sim
        self._pause_sim(False)

        # Moving the agent
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Leave running until next observation comes in
        self.new_obs_event = False
        while not self.new_obs_event:
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # Pause the simulation after the next observation comes in
        self._pause_sim(True)

        state, terminated = self.process_obs(self.latest_obs)
        
        reward = self.process_reward(action, state, terminated)

        truncated = False # Used if we ran maximum number of steps.
        return state, reward, terminated, truncated, {}

    
    ## HELPER FUNCTIONS
    def process_action(self, action: int):
        '''
        @brief Translate the action the robot is taking into a Twist message
        @param action: an integer where 0 is left, 1 is forward and 2 is right 
        @retval vel_cmd: Twist message equivalent of the action
        '''
        vel_cmd = RosTwist()

        if action == 0: # Left
            vel_cmd.linear.x = DRIVE_WHILE_TURN_SPEED
            vel_cmd.angular.z = TURN_SPEED

        elif action == 1: # Forward
            vel_cmd.linear.x = DRIVE_SPEED
            vel_cmd.angular.z = 0.0

        elif action == 2: # Right
            vel_cmd.linear.x = DRIVE_WHILE_TURN_SPEED
            vel_cmd.angular.z = -TURN_SPEED
        
        return vel_cmd


    def process_reward(self, action, state, terminated):
        '''
        @brief depending on the action, state and terminated values assign
               a reward
        @retval reward: the reward for the current step
        '''
        if terminated:
            reward = -1000

        elif state == 1: # The robot is aligned with the line
            if action == 1:
                reward = 6 # Extra reward if robot is driving straight and aligned with the line. 

            else:
                reward = 5

        else:
            reward = 1 # The robot is not aligned with the line
        
        return reward


    def process_obs(self, obs: RosImage):
        '''
        @brief Identifies the line in the image and populates the state array
               based on the line position (i.e. state=0 line is left, 
               state=1 line is in the center)
        @param obs raw image from Gazebo sim
        @retval state: the line position as a value: 
            -1 (no line), 0 (left), 1 (center), 2 (right)
        @retval terminated: the termination condition occured - the line is no 
                    longer in the field of the camera
        '''
        state = 0
        terminated = False
        
        cv_image = CvBridge().imgmsg_to_cv2(obs, desired_encoding='bgr8')

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) # Grayscales the image

        _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV) # Makes all pixels with brightness below 70 white and the rest black to make the line stand out

        height, width = binary.shape # Find image dimensions

        left = binary[:, :width // 3] # splitting image into 3 sections
        center = binary[:, width // 3:2 * width // 3]
        right = binary[:, 2 * width // 3:]

        left_count = cv2.countNonZero(left) # Counts the pixels of the line in each frame
        center_count = cv2.countNonZero(center)
        right_count = cv2.countNonZero(right)

        # Determine if the line is visible
        counts = [left_count, center_count, right_count]

        if max(counts) < 10: # Buffer for noise
            state = -1
            terminated = True # Terminates the program if the line is no longer in the image (learned what not to do)
        else:
            state = int(np.argmax(counts)) # The state is the count with the highest number of pixels

        return state, terminated


    # Shutting down ROS2
    def user_close(self):
        self.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()