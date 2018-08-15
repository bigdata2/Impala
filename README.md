# Impala
This is an implementation of distributed deep reinforcement learning framework called [Importance-Weighted Actor-Learner Architectures (Impala)](https://deepmind.com/blog/impala-scalable-distributed-deeprl-dmlab-30/) Using DeepMind Lab [simulator](https://github.com/deepmind/lab) and distributed execution framework [Ray](https://github.com/ray-project/ray). Currently, only 1-step importance sampling is implemented, V-trace will be implemented later.
To run the program (actor/learner/parameter server) use the following command: python learner.py --length 100 --actors 10 -s --level_script <level script such as seekavoid_arena_01>
Set IP address of RAY HEAD in learner.py before running.
