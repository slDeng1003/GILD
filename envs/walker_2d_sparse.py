import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class SparseWalker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,sparse_val = 1.):
        self._current_step = 0
        self._max_episode_steps = 1000
        self.sparse_val = sparse_val
        print('Sparse val', self.sparse_val)
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self._current_step += 1
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]


        # --------- Sparse Reward ---------
        if (posafter - self.init_qpos[0]) > self.sparse_val:
            alive_bonus = 1.0
            reward = (posafter - posbefore) / self.dt
            reward += alive_bonus
            reward -= 1e-3 * np.square(a).sum()
        else:
            reward = 0.
        
        s = self.state_vector()
        done = not (np.isfinite(s).all() and height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        if self._current_step >= self._max_episode_steps:
            done = True

        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self._current_step = 0 # reset the step
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20