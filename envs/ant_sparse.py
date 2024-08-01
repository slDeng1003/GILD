import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class SparseAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, saprse_val = 1.):
        self._current_step = 0
        self._max_episode_steps = 1000
        self.sparse_val = saprse_val
        print('Sparse val', self.sparse_val)
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self._current_step += 1
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone

        # --------- Sparse Reward ---------
        if xposafter - self.init_qpos[0] > self.sparse_val:
            forward_reward = (xposafter - xposbefore) / self.dt
            ctrl_cost = 0.5 * np.square(a).sum()
            contact_cost = (
                0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            )
            survive_reward = 1.0
            reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        else:
            reward=0.

        if self._current_step >= self._max_episode_steps:
            done = True

        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self._current_step = 0 # reset the step
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5