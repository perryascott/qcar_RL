#!/usr/bin/env python3


import numpy as np

import time

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import pickle

import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()


# Define the Ornstein Uhlenbeck process
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) *
            np.random.normal(size=self.mean.shape)
        )

        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

# Setup buffer
class Learner:
    def __init__(self, buffer_name, network_name, new_network, buffer_capacity=100000, batch_size=64, num_states=2, num_actions=2):

        self.network_name = network_name
        self.buffer_capacity = buffer_capacity

        self.batch_size = batch_size
        self.num_states = num_states
        self.num_actions = num_actions

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

        self.dbfile = open(buffer_name, 'rb')

        self.buffer_counter = 0

        while(True):
            try:
                buffer_list = pickle.load(self.dbfile)
                for i in range(len(buffer_list)):

                    if(i % 4 == 0):
                        list_length = np.shape(buffer_list[i])[0]
                        self.state_buffer[self.buffer_counter:self.buffer_counter +
                                          list_length, :] = buffer_list[i]
                    elif(i % 4 == 1):
                        self.action_buffer[self.buffer_counter:self.buffer_counter +
                                           list_length] = buffer_list[i]
                    elif(i % 4 == 2):
                        self.reward_buffer[self.buffer_counter:self.buffer_counter +
                                           list_length, :] = buffer_list[i]
                    else:
                        self.next_state_buffer[self.buffer_counter:self.buffer_counter +
                                               list_length, :] = buffer_list[i]

                    print(
                        "i is " + str(i) + " shape of buffer_list[i] is " + str(np.shape(buffer_list[i])))

                self.buffer_counter = self.buffer_counter + list_length

            except:

                break

        self.buffer_counter = self.buffer_counter - 1


        # set bounds
        self.throttle_upper_bound = .25  # goes up to .3
        self.throttle_lower_bound = -.25  # goes down to -.3

        self.steering_upper_bound = .5
        self.steering_lower_bound = -.5

        self.tau = .005
        self.gamma = 0.95
        critic_lr = 0.0002
        actor_lr = 0.0001

        print("Critic Learning Rate ->  {}".format(critic_lr))
        print("Actor Learning Rate ->  {}".format(actor_lr))
        print("Tau ->  {}".format(self.tau))

        print("Max Value of throttle ->  {}".format(self.throttle_upper_bound))
        print("Min Value of throttle ->  {}".format(self.throttle_lower_bound))

        print("Max Value of steering ->  {}".format(self.steering_upper_bound))
        print("Min Value of steering ->  {}".format(self.steering_lower_bound))

        # create target and model networks
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        if(new_network == False):
            self.actor_model.load_weights(
                self.network_name + str("/actor_model.h5"))
            self.critic_model.load_weights(
                self.network_name + str("/critic_model.h5"))
            self.target_critic.load_weights(
                self.network_name + str("/target_critic.h5"))
            self.target_actor.load_weights(
                self.network_name + str("/target_actor.h5"))

        self.actor_loss_list = []
        self.critic_loss_list = []

    # Enable reward calculation with collected buffer
    def calc_reward(self):
        for i in range(self.buffer_counter + 1):
            v = self.next_state_buffer[i, 0]
            self.reward_buffer[i] = -abs(abs(v) - 1.0)

    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Update actor and critic loss
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:

            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * \
                self.target_critic(
                    [next_state_batch, target_actions], training=True)
            critic_value = self.critic_model(
                [state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(
            critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model(
                [state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(
                actor_loss, self.actor_model.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor_model.trainable_variables))

        return critic_loss, actor_loss

    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(
            self.next_state_buffer[batch_indices])

        crit_loss, act_loss = self.update(
            state_batch, action_batch, reward_batch, next_state_batch)
        self.update_target(self.target_actor.variables,
                           self.actor_model.variables, self.tau)
        self.update_target(self.target_critic.variables,
                           self.critic_model.variables, self.tau)

        self.actor_loss_list.append(act_loss.numpy())
        self.critic_loss_list.append(crit_loss.numpy())

    # Slowly update target weights
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(8, activation="relu")(inputs)
        out = layers.Dense(8, activation="relu")(out)
        outputs = layers.Dense(self.num_actions, activation="tanh",
                               kernel_initializer=last_init)(out)


        # need to actually multiply each output by the throttle and steer bounds respectively
        output_throttle = layers.Dense(1, activation="tanh",
                                       kernel_initializer=last_init)(out)
        output_throttle = output_throttle * self.throttle_upper_bound

        outputs = output_throttle

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

        out = layers.Dense(8, activation="relu")(concat)
        out = layers.Dense(8, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    # Define a function that adds noise to the action
    def policy_w_noise(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(state))
        #noise = noise_object()

        noise = np.random.normal(0, .02)

        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        # print(type(sampled_actions))
        legal_action_vel = np.clip(
            sampled_actions, self.throttle_lower_bound, self.throttle_upper_bound)
        return [np.squeeze(legal_action_vel)]

    # Define a function without noise added to the action
    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))

        # Adding noise to action
        sampled_actions = sampled_actions.numpy()

        # We make sure action is within bounds
        # print(type(sampled_actions))
        legal_action_vel = np.clip(
            sampled_actions, self.throttle_lower_bound, self.throttle_upper_bound)
        return [np.squeeze(legal_action_vel)]

    # Save trained models
    def save_models(self):
        self.actor_model.save_weights(
            self.network_name + str("/actor_model.h5"))
        self.critic_model.save_weights(
            self.network_name + str("/critic_model.h5"))
        self.target_critic.save_weights(
            self.network_name + str("/target_critic.h5"))
        self.target_actor.save_weights(
            self.network_name + str("/target_actor.h5"))


def main():

    n_a = 2
    n_s = 0
    action_space = 1
    state_space = 1

    # environment characteristics
    num_states = state_space + n_s + n_a * action_space
    num_actions = action_space

    learning_index = 1

    # MAKE SURE THIS IS FALSE IF BUILDING ON EXISTING NETWORK
    new_network = True
    learning_iterations = 50000

    # hyperparams
    noise_std = .05

    buffer_name = "./buffer_data/ref_vel_buffer_1"
    network_name = "./network_data/ref_vel_network_06"

    learner = Learner(buffer_name, network_name, new_network,
                      20000, 64, num_states, num_actions)

    print(learner.buffer_counter)
    print(learner.state_buffer)
    print(learner.action_buffer)

    decay_rate = .96
    decay_steps = 25000
    while(True):

        # make buffer large enough before starting to train
        if(learning_index < learning_iterations):
            learner.learn()
            print(learning_index)

            # Plot actor loss and critic loss
            if(learning_index % 10000 == 0):
                plt.plot(learner.critic_loss_list, label="critic")
                plt.ylabel('loss')
                plt.xlabel('training iteration')
                plt.title("crtiic loss vs training iterations")

                plt.plot(learner.actor_loss_list, label="actor")
                plt.ylabel('loss')
                plt.xlabel('training iteration')
                plt.title("actor loss vs training iterations")

                plt.legend(loc="upper left")
                plt.show()

        else:
            learner.save_models()

            break

        learning_index = learning_index + 1

    plt.plot(learner.critic_loss_list, label="critic")
    plt.ylabel('loss')
    plt.xlabel('training iteration')
    plt.title("crtiic loss vs training iterations")

    plt.plot(learner.actor_loss_list, label="actor")
    plt.ylabel('loss')
    plt.xlabel('training iteration')
    plt.title("actor loss vs training iterations")

    plt.legend(loc="upper left")
    plt.show()


if __name__ == '__main__':
    main()
