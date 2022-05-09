#!/usr/bin/env python3

from __future__ import division, print_function, absolute_import

import roslib
import rospy
import numpy as np
from qcar.product_QCar import QCar
from qcar.q_interpretation import *

from std_msgs.msg import String, Float32
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from sensor_msgs.msg import BatteryState
import time

import tensorflow as tf
from tensorflow.keras import layers


import numpy as np
import pickle

class GetPolicy:
	def __init__(self, network_name, num_states=2, num_actions=2):



		self.network_name = network_name
	

	
		self.num_states = num_states
		self.num_actions = num_actions
		
		
		#set bounds
		self.throttle_upper_bound = .25 #goes up to .3
		self.throttle_lower_bound = -.25 #goes down to -.3

		self.steering_upper_bound = .5
		self.steering_lower_bound = -.5

		#create target and model networks
		self.actor_model = self.get_actor()
		self.critic_model = self.get_critic()


		
		self.actor_model.load_weights(self.network_name + str("/actor_model.h5"))
		self.critic_model.load_weights(self.network_name + str("/critic_model.h5"))

	def get_actor(self):
		# Initialize weights between -3e-3 and 3-e3
		last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

		inputs = layers.Input(shape=(self.num_states,))
		out = layers.Dense(12, activation="relu")(inputs)
		out = layers.Dense(12, activation="relu")(out)
		outputs = layers.Dense(self.num_actions, activation="tanh",
                           kernel_initializer=last_init)(out)

		# Our upper bound is 2.0 for Pendulum.

		#need to actually multiply each output by the throttle and steer bounds respectively
		output_throttle = layers.Dense(1, activation="tanh",
                           kernel_initializer=last_init)(out)
		output_throttle = output_throttle * self.throttle_upper_bound
		
		output_steering = layers.Dense(1, activation="tanh",
                           kernel_initializer=last_init)(out)
		
		output_steering = output_steering * self.steering_upper_bound
		outputs =  layers.Concatenate()([output_throttle, output_steering])

		

		#outputs = outputs * np.array([self.steering_upper_bound, self.throttle_upper_bound])
		model = tf.keras.Model(inputs, outputs)
		return model


	def get_critic(self):
		# State as input
		state_input = layers.Input(shape=(self.num_states))
		state_out = layers.Dense(8, activation="relu")(state_input)
		state_out = layers.Dense(8, activation="relu")(state_out)

		# Action as input
		action_input = layers.Input(shape=(self.num_actions))
		action_out = layers.Dense(8, activation="relu")(action_input)

		# Both are passed through seperate layer before concatenating
		concat = layers.Concatenate()([state_out, action_out])

		out = layers.Dense(12, activation="relu")(concat)
		out = layers.Dense(12, activation="relu")(out)
		outputs = layers.Dense(1)(out)

		# Outputs single value for give state-action
		model = tf.keras.Model([state_input, action_input], outputs)

		return model

	def policy_w_noise(self, state, noise_object):
		sampled_actions = tf.squeeze(self.actor_model(state))
		#noise = noise_object()

		noise = np.random.normal(0,.02)

		# Adding noise to action
		sampled_actions = sampled_actions.numpy() + noise

		# We make sure action is within bounds
		#rospy.loginfo(type(sampled_actions))
		legal_action_vel = np.clip(sampled_actions, self.throttle_lower_bound, self.throttle_upper_bound)
		return [np.squeeze(legal_action_vel)]

	def policy(self, state):
		sampled_actions = tf.squeeze(self.actor_model(state))
		

		# Adding noise to action
		sampled_actions = sampled_actions.numpy()

		# We make sure action is within bounds
		#rospy.loginfo(type(sampled_actions))
		legal_action_vel = np.clip(sampled_actions[0], self.throttle_lower_bound, self.throttle_upper_bound)
		legal_action_str = np.clip(sampled_actions[1], self.steering_lower_bound, self.steering_upper_bound)
		return [legal_action_vel,legal_action_str]


class Buffer:
	def __init__(self, buffer_name, new_buffer = False, buffer_capacity=100000, num_states=2, num_actions=2):
		
		self.buffer_name = buffer_name
		self.new_buffer = new_buffer


		self.buffer_capacity = buffer_capacity
		self.num_states = num_states
		self.num_actions = num_actions

		
		self.buffer_counter = 0

		self.state_buffer = np.zeros((self.buffer_capacity, num_states))
		self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
		self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

		
		if(self.new_buffer):
			self.dbfile = open(self.buffer_name, 'wb')
		else:
			self.dbfile = open(self.buffer_name, 'ab')
			


	def record(self, obs_tuple):
		# Set index to zero if buffer_capacity is exceeded,
		# replacing old records
		index = self.buffer_counter % self.buffer_capacity

		self.state_buffer[index] = obs_tuple[0]
		self.action_buffer[index] = obs_tuple[1]
		self.next_state_buffer[index] = obs_tuple[2]

		self.buffer_counter += 1

	def save_buffer(self):

		
		buffer_list = [self.state_buffer, self.action_buffer, self.next_state_buffer]
		pickle.dump(buffer_list, self.dbfile)                     
		self.dbfile.close()
		
		return



class Buffer_Node(object):

	def __init__(self):
		super().__init__()
		
		rospy.on_shutdown(self.shutting_down)

			
		self.n_a = 0
		self.n_s = 0
		self.action_space = 2
		self.state_space = 6


		#environment characteristics
		self.num_states = self.state_space
		self.num_actions = self.action_space


		#state action values

		self.state = np.zeros(self.num_states)
		self.prev_state = self.state

		self.action = np.array([0.,0.])
		self.actions = np.empty((0,2))
		self.recent_action = False



		#position objects
		self.x = 0
		self.z = 0
		self.yaw = 0
		self.time = 0

		self.x_prev = 0
		self.z_prev = 0
		self.yaw_prev = 0
		self.time_prev = 0

		#state and error variables
		self.yaw = 0
		self.radial_angle = 0
		self.heading_error = 0

		self.radius = 0
		self.radial_error = 0

		self.speed = 0
		self.speed_error = 0

		#reference values
		self.ref_radius = 1.143
		self.ref_speed = .7


		#subscribe to position and controller command
		self.pose_sub_ = rospy.Subscriber('/mocap/rigid_bodies/RigidBody_01/pose', PoseStamped, self.update_coords, queue_size=100)
		self.cmd_sub_ = rospy.Subscriber('/qcar/user_command', Vector3Stamped, self.process_cmd, queue_size=10)
		self.cmd_pub_ = rospy.Publisher('/critic_value', Vector3Stamped, queue_size=100)

	


		self.sampling_rate = rospy.Rate(5) #Hz

		self.state_mean =[ 0.08157198, -0.01861664, -0.02135922,  0.00041671, -0.02958437,0.00024513]
		self.state_variance = [6.54200947e-02, 6.61160108e-01, 6.51017401e-03, 2.04472943e-02, 9.38314797e-01, 3.46092837e+01]		
		#self.action_mean = [.125, 
		#self.action_variance = [0.00623129, 0.03280919]
		

		#MAKE SURE THIS IS FALSE IF ADDING TO BUFFER
		self.new_buffer = False

		#create buffere and noise objects
		buffer_name = "./buffer_data/ref_path_buffer_2"
		network_name = "./network_data/ref_path2_network5"
		self.records_per_run = 1000
		
		self.buffer = Buffer(buffer_name, self.new_buffer, self.records_per_run , self.num_states, self.num_actions)	
		self.getPolicy = GetPolicy(network_name, self.num_states, self.num_actions)

		
		start_time = rospy.Time.now()
		self.start_time = start_time.secs + start_time.nsecs*10**(-9)

#-------------------------------------------------------------------------------------------------
	def looping(self):	


		while not rospy.is_shutdown():
			self.sampling_rate.sleep()
			#calculate speed or change in distance
			
			curr_time = rospy.Time.now()
			self.time = curr_time.secs + curr_time.nsecs*10**(-9)
			
			#get average of acitons taken in last .2 seconds
			if(self.recent_action):
				self.action = np.mean(self.actions, axis=0)

			self.recent_action = False
			self.actions = np.empty((0,2))

			#calculate speed and speed error
			self.speed = np.sqrt((self.x-self.x_prev)**2 + (self.z-self.z_prev)**2)*5
			self.speed_error = self.speed - self.ref_speed
			self.x_prev = self.x
			self.z_prev = self.z



			#get errors
			self.state[0] = self.speed_error
			self.state[2] = self.radial_error
			self.state[4] = self.heading_error

			#get error rates
			self.state[1] = (self.state[0] - self.prev_state[0]) * 5
			self.state[3] = (self.state[2] - self.prev_state[2]) * 5
			self.state[5] = (self.state[4] - self.prev_state[4]) * 5

			self.prev_state = self.state.copy()


			#make buffer large enough before starting to train
			if(self.buffer.buffer_counter < self.records_per_run):
				self.buffer.record((self.prev_state, self.action, self.state))

				
				rospy.loginfo("My action is " + str(self.action))
				rospy.loginfo("My state is " + str(self.state))


				norm_state = (self.state - self.state_mean)/np.sqrt(self.state_variance)
				
				tf_prev_state = tf.expand_dims(tf.convert_to_tensor(norm_state), 0)
				action = self.getPolicy.policy(tf_prev_state)
				
				rospy.loginfo("Network recommended action " + str(action))

				
				tf_action= tf.expand_dims(tf.convert_to_tensor(action), 0)


				critic_value = self.getPolicy.critic_model([tf_prev_state, tf_action])
				rospy.loginfo("Critic Value at current state = " + str(critic_value))
				#comment this line out to manually train
				self.send_command(action[0],action[1], critic_value.numpy())

			else:
				self.buffer.save_buffer()
				rospy.loginfo("done and buffer saved")
				rospy.signal_shutdown("buffer collection done")

			
								
	def process_cmd(self, sub_cmd):
		vel_cmd = sub_cmd.vector.x
		str_cmd = sub_cmd.vector.y
		command = np.array([[vel_cmd, str_cmd]])

		self.actions = np.append(self.actions, command, axis=0)
		self.recent_action = True
		return	

	def send_command(self, vel, ang, value):
		pub_cmd = Vector3Stamped()
		pub_cmd.header.stamp = rospy.Time.now() 
		pub_cmd.header.frame_id = 'command_input'
		pub_cmd.vector.x = vel
		pub_cmd.vector.y = ang
		pub_cmd.vector.z = value

		command = np.array([[vel, ang]])

		self.actions = np.append(self.actions, command, axis=0)
		self.recent_action = True

		self.cmd_pub_.publish(pub_cmd)
	
	
		
	def update_coords(self,pose_msg):


		self.yaw_prev = self.yaw

		self.x = pose_msg.pose.position.x
		self.z = pose_msg.pose.position.z

		# calculate yaw angle from quaternion
		x1 = pose_msg.pose.orientation.x
		y1 = pose_msg.pose.orientation.y
		z1 = pose_msg.pose.orientation.z
		w1 = pose_msg.pose.orientation.w
		self.yaw = math.atan2(2*x1*z1-2*w1*y1,-1*(1-2*x1**2-2*y1**2))

		#calculate radial angle and heading error
		p1x = -.2115
		p1z = 1.3277
		p2x = self.x
		p2z = self.z

		delta_x = p1x - p2x
		delta_z = p2z - p1z
		theta1 = math.atan2(-delta_x, delta_z)

		theta1 = theta1 + 3.141592 / 2
		if(theta1 > 3.141592):
			theta1 = theta1 - 2*3.141592 
		self.radial_angle = theta1

		#if(abs(self.yaw) > 3 and self.radial_angle*self.yaw < 0):
		if(self.radial_angle*self.yaw < 0):

			if(self.yaw < 0):
				self.yaw = self.yaw + 2*3.141592
			else:
				self.radial_angle = self.radial_angle + 2*3.141592
		self.heading_error = self.yaw - self.radial_angle
		#rospy.loginfo("Headiner error is " + str(self.heading_error))


		#calculate radius and radial error
		self.radius = np.sqrt(delta_x**2 + delta_z**2)
		self.radial_error = self.radius - self.ref_radius
		
		

	def shutting_down(self):
  		print("shutdown time!")



if __name__ == '__main__':
	rospy.init_node('buffer_node')
	r = Buffer_Node()
	r.looping()
	rospy.spin()
	
	