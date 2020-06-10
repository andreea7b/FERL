# FERL: Feature Expansive Reward Learning

Control, planning, and learning system for human-robot interaction with a JACO2 7DOF robotic arm. Supports learning rewards from physical corrections, as well as learning new reward features.

## Dependencies
* Ubuntu 14.04, ROS Indigo, OpenRAVE, Python 2.7
* or_trajopt, or_urdf, or_rviz, prpy, pr_ordata
* kinova-ros
* fcl

## Running the FERL System on a Robot Arm
### Setting up the JACO2 Robot
Turn the robot on and put it in home position by pressing and holding the center (yellow) button on the joystick.

In a new terminal, turn on the Kinova API by typing:
```
roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=j2s7s300 use_urdf:=true
```

### Feature Expansive Reward Learning
To demonstrate FERL on the Jaco arm, run (in another terminal window):
```
roslaunch FERL feature_elicitator.launch
```
The launch file first reads the corresponding yaml `config/feature_elicitator.yaml` containing all important parameters, then runs `feature_elicitator.py`. Given a start, a goal, and other task specifications, a planner plans an optimal path, then the controller executes it. For a selection of planners and controllers, see `src/planners` (TrajOpt supported currently) and `src/controllers` (PID supported currently). The yaml file should contain parameter information to instantiate these two components.

A human can apply a physical correction to change the way the robot is executing the task. Depending on the learning method used, the robot learns from the human torque accordingly and updates its trajectory in real-time. With FERL, if the robot has low confidence in its ability to explain the human correction, it enters a feature learning stage: the robot stops following the planned trajectory and, instead, listens for feature traces from the human. Once it learns a new feature from these traces, the robot updates its reward, and resumes execution of the new path, from the position where it paused.

Some important parameters for specifying the task in the yaml include:
* `start`: Jaco start configuration
* `goal`: Jaco goal configuration
* `goal_pose`: Jaco goal pose (optional)
* `T`: Time duration of the path
* `timestep`: Timestep dicretization between two consecutive waypoints on a path.
* `feat_list`: List of features the robot's initial internal representation contains. Options: "table" (distance to table), "coffee" (coffee cup orientation), "human" (distance to human), "laptop" (distance to laptop), "proxemics" (distance from the human, more penalized in front than on the sides), "efficiency" (velocity), "between objects" (distance from between two objects).
* `feat_weights`: Initial feature weights.
* `CONFIDENCE_THRESHOLD`: Threshold for confidence estimate below which the
  algorithm enters feature learning mode.
* `N_QUERIES`: Number of feature traces to ask a person for per learned feature.
* `nb_layers`: Number of layers for a newly learned feature.
* `nb_units`: Number of units for a newly learned feature.

Some task-specific parameters in addition to the ones above include
environment, planner, controller, and reward learner specific parameters that have been tuned and
don't need to be changed. If you have questions about how to change those,
contact abobu<at>berkeley.edu.

### References
* TrajOpt Planner: http://rll.berkeley.edu/trajopt/doc/sphinx_build/html/index.html
* PID Control Reference: https://w3.cs.jmu.edu/spragunr/CS354/handouts/pid.pdf

## Running the FERL Feature Learning in a Docker container

### Starting the Docker Container
We also provide a docker image with Ubuntu, ROS, and all necessary packages installed. Just clone the FERL repo and then run the following docker command to download and run the docker image.

`docker run -it -p 8888:8888 -v <full path to cloned FERL repo>:/root/catkin_ws/src/FERL mariuswi/trajopt_ws:1.0`

This will bring you to the command line of the docker container with port forwarding to 8888 activated (e.g. for running the example notebooks).

### Running the Example Notebooks
We provide two example notebooks both for our Method FERL, and for our baseline ME-IRL. The notebooks walk you through the reward learning process, for FERL: loading traces, training & examining a learned feature, and combining it with the known ones to a final reward, and for ME-IRL: loading demonstrations, and running maximum entropy IRL to learn a reward function directly from state.

To run them and explore our method, just run the docker image. Start jupyter in the container with:

`jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root`

Navigate to src/example_notebooks, and have fun interactively exploring FERL.
