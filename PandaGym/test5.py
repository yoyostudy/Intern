import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance

from panda_gym.pybullet import PyBullet


class MyTask(Task):
    def __init__(self, sim):
        super().__init__(sim)
        # create an cube
        self.sim.create_box(body_name="object", half_extents=np.array([1, 1, 1]), mass=1.0, position=np.array([0.0, 0.0, 0.0]))

    def reset(self):
        # randomly sample a goal position
        self.goal = np.random.uniform(-10, 10, 3)
        # reset the position of the object
        self.sim.set_base_pose("object", position=np.array([0.0, 0.0, 0.0]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))

    def get_obs(self):
        # the observation is the position of the object
        observation = self.sim.get_base_position("object")
        return observation

    def get_achieved_goal(self):
        # the achieved goal is the current position of the object
        achieved_goal = self.sim.get_base_position("object")
        return achieved_goal

    def is_success(self, achieved_goal, desired_goal, info={}):  # info is here for consistancy
        # compute the distance between the goal position and the current object position
        d = distance(achieved_goal, desired_goal)
        # return True if the distance is < 1.0, and False otherwise
        return np.array(d < 1.0, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info={}):  # info is here for consistancy
        # for this example, reward = 1.0 if the task is successfull, 0.0 otherwise
        return self.is_success(achieved_goal, desired_goal, info).astype(np.float32)
    

sim = PyBullet(render_mode="human")
task = MyTask(sim)

task.reset()
print(task.get_obs())
print(task.get_achieved_goal())
print(task.is_success(task.get_achieved_goal(), task.get_goal()))
print(task.compute_reward(task.get_achieved_goal(), task.get_goal()))