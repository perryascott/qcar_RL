#!/usr/bin/env python3


import numpy as np

import time

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import pickle

import matplotlib.pyplot as plt

# Setup buffer
class Learner:
    def __init__(self, buffer_name, network_name, new_network, buffer_capacity=100000, batch_size=64, num_states=2, num_actions=2):

        self.network_name = network_name
        self.buffer_capacity = buffer_capacity

        self.batch_size = batch_size
        self.num_states = num_states
        self.num_actions = num_actions

        # read in buffer from file
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

        mean1 = np.mean(self.state_buffer, axis=0)

        var1 = np.var(self.state_buffer, axis=0)

        self.state_stats = [mean1, var1]

        # Calculate mean and variance for normalization
        mean2 = np.mean(self.reward_buffer)
        var2 = np.var(self.reward_buffer)
        self.reward_stats = [mean2, var2]

        mean3 = np.mean(self.action_buffer, axis=0)
        var3 = np.var(self.action_buffer, axis=0)
        self.action_stats = [mean3, var3]

        # set bounds
        self.throttle_upper_bound = .25  # goes up to .3
        self.throttle_lower_bound = -.25  # goes down to -.3

        self.steering_upper_bound = .5
        self.steering_lower_bound = -.5

        self.tau = .005
        self.gamma = 0.95
        self.critic_lr = 0.002 / 100
        self.actor_lr = 0.003 / 100

        print("Critic Learning Rate ->  {}".format(self.critic_lr))
        print("Actor Learning Rate ->  {}".format(self.actor_lr))
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

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

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
        #print("Critic Loss = " + str(crit_loss.numpy()))
        #print("Actor Loss = " + str(act_loss))

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(24, activation="relu")(inputs)
        out = layers.Dense(24, activation="relu")(out)
        outputs = layers.Dense(self.num_actions, activation="tanh",
                               kernel_initializer=last_init)(out)

        # need to actually multiply each output by the throttle and steer bounds respectively
        output_throttle = layers.Dense(1, activation="tanh",
                                       kernel_initializer=last_init)(out)
        output_throttle = output_throttle * self.throttle_upper_bound

        output_steering = layers.Dense(1, activation="tanh",
                                       kernel_initializer=last_init)(out)

        output_steering = output_steering * self.steering_upper_bound
        outputs = layers.Concatenate()([output_throttle, output_steering])

        #outputs = outputs * np.array([self.steering_upper_bound, self.throttle_upper_bound])
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.num_states))
        state_out = layers.Dense(24, activation="relu")(state_input)
        state_out = layers.Dense(24, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(24, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(24, activation="relu")(concat)
        out = layers.Dense(24, activation="relu")(out)
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

    # Define a function without added noise to the action
    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))

        # Adding noise to action
        sampled_actions = sampled_actions.numpy()

        # We make sure action is within bounds
        # print(type(sampled_actions))
        legal_action_vel = np.clip(
            sampled_actions, self.throttle_lower_bound, self.throttle_upper_bound)
        return [np.squeeze(legal_action_vel)]

    # Save trained weights
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
    action_space = 2
    state_space = 3

    # environment characteristics
    num_states = state_space + n_s + n_a * action_space
    num_actions = action_space

    learning_index = 1

    # MAKE SURE THIS IS FALSE IF BUILDING ON EXISTING NETWORK
    new_network = True
    learning_iterations = 300000

    # hyperparams
    noise_std = .05

    buffer_name = "./buffer_data/ref_path_buffer_1"
    network_name = "./network_data/ref_path_network_4"

    learner = Learner(buffer_name, network_name, new_network,
                      20000, 256, num_states, num_actions)
    noise_object = OUActionNoise(mean=np.zeros(
        1), std_deviation=float(noise_std) * np.ones(1))

    print(learner.buffer_counter)
    print(learner.state_buffer)
    print(learner.action_buffer)

    decay_rate = .96
    decay_steps = 25000
    while(True):

        # make buffer large enough before starting to train
        if(learning_index < learning_iterations):
            if(learning_index % decay_steps == 5000):
                learner.critic_lr = learner.critic_lr * decay_rate
                learner.actor_lr = learner.actor_lr * decay_rate
                learner.actor_optimizer = tf.keras.optimizers.Adam(
                    learner.actor_lr)
                learner.critic_optimizer = tf.keras.optimizers.Adam(
                    learner.critic_lr)

            if(learning_index % decay_steps == 5000):
                # Plot losses
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

            learner.learn()
            print(learning_index)

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
