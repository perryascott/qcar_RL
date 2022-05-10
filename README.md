
# Autonomous Driving with Reinforcement Learning 

This project implements the DDPG algorithm to a scaled car, aiming to train it for a constant-velocity tracking and a circular path tracking tasks.

![](Readme_file/QCar.jpg)

## Algorithm Description

DDPG is the abbreviation of the Deep Deterministic Policy Gradient algorithm. It is a modification based on the deep Q-network (DQN) and the deterministic policy gradient (DPG) algorithms, such that it is able to use neural network function approximators to learn action functions and policies in continuous observation and action spaces. It also uses a replay buffer and a delayed update target network to deal with the unstable feature of the nonlinear function approximations via neural networks. Detailed explanations of DDPG is included in our report. We also refer readers to read the original papers of DDPG, DQN, and DPG to gain insights.

## File Description

### Buffer Data

Several tests were done with hand controlled episodes, and transition information was recorded in the replay buffer data. The buffer data was used to train several policy and critic networks located in the network_data folder.

The buffer files labeled "ref_path..." were used in the task of trying to follow the circle path in the room. 

Files labeled "ref_vel..." were used in the reference velocity task.

### Network Data

Trained network weights. All the existing network files are for the reference velocity task.

### Bag Data

All these files contain ROStopic information which contains information about vehicle states, actions, and critic and policy network outputs in real time.

### Buffer Node

This is a python-ROS script which was run on the physical vehicle using SSH to collect the replay buffer in real-time. Additionally, offline trained networks could be moved onto the vehicle computer and tested using this script as well as for gathering online training data. Due to the nature of the system, you will need a RC car which runs ROS-Python to run this script.

### Learning Node

This file was used to train policy and critic network using DDPG. It utilizes the chosen buffer data to run the script. Buffer rewards can be recalculated using this script as well.

### Learning Node Vel

This file is similar to the learning node file but more problem specific for the velocity task.

### Testing

This is a testing file to show some results without application to the RC car. Learning_node file can be ran and the critic and actor loss for every 10000 learning iterations will be shown.

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)