# qcar_RL

The following project involved using a reinforcement learning algorithm, deep deterministic policy gradient (DDPG) to learn several tasks for a physical scaled autonomous vehicle called the Qcar. Several tests were done with hand controlled episodes and state and action information was recorded in the buffer data file. The buffer data was used to train several policy and critic networks located in the network_data folder.

Buffer data:
-the buffer files labeled "ref_path..." were used in the task of trying to the follow the circle path in the room. 
-files labeled "ref_vel..." were used in the reference velocity task.

network data:
-all the existing network files are for the reference velocity task

bag data:
-all these files contain ROStopic information which contains info about vehicle states, actions, and critic and policy network outputs in real time.

buffer_node:
This is a python-ROS script which was run on the physical vehicle using SSH to collect state information in real-time. Additionally, offline trained networks could be moved onto the vehicle computer and tested using this script as well as for gathering online training data. Due to the nature of the system, you will need a RC car which runs ROS-Python to run this script

learning_node:
Used to train policy and critic network using DDPG. Utilizes the chosen buffer data to run the script. Buffer rewards can be recalculated using this script as well.

learning_node_vel:
Similar to learning node but more problem specific for the velocity task

readData.m:
used to read and plot bagData captured with ROS in real time.

Testing:
learning_node can be ran and the critic and actor loss every 10000 learning iterations will be shown.