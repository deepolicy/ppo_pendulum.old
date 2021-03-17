import tensorflow as tf
import numpy as np
from tqdm import trange
from baselines.common.tf_util import get_session, initialize
from baselines.common.running_mean_std import RunningMeanStd
import gym

from gym_goal_platform.utils import scale_to
from .policy import Policy
from .value_net import value_nn

class Sampler(object):
    def __init__(self, obs_norm):

        self.env = gym.make('Pendulum-v0')

        self.rms = {
            'reward': RunningMeanStd(epsilon=1e-9, shape=(1,)),
        }

        self.ep_total_reward_list = []
        
        self.obs_norm = obs_norm

        self.build_network()
    
        # # Create session
        # self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())

        # baselines.common.tf_util
        self.sess = get_session()
        initialize()

    def build_network(self):

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):

            obs_dim = 3
            self.policy = Policy(obs_dim=obs_dim)

            self.value = value_nn(self.policy.obs_ph, obs_dim=obs_dim)

    def get_act_val_nlogp(self, obs):

        obs -= self.obs_norm['mean']
        obs /= self.obs_norm['std']

        assert np.all(obs > -3.)
        assert np.all(obs < 3.)

        fetches = {
            "act": [self.policy.action_dist[i] for i in [0]],
            'value': self.value,
            'neglogpacs': [self.policy.action_dist[i] for i in [1]],
        }

        feed_dict = {
            self.policy.obs_ph: [obs],
        }

        results = self.sess.run(fetches, feed_dict)

        # print('results')
        # print(results)
        '''
        results
        {'act': [array([[-0.11286727]], dtype=float32)], 'value': array([[-0.24949437]], dtype=float32), 'neglogpacs': [array([0.9240459], dtype=float32)]}
        '''

        action = results['act'][0][0]
        action = scale_to(action, -2., 2., 1.5)

        return action, results, obs

    def run_env(self, sample_length=100):

        env = self.env

        sample_data = []

        while 1:
            state = env.reset()
            ep_total_reward = 0

            while 1:
                action, act_val_nlogp, state_norm = self.get_act_val_nlogp(state)

                # time.sleep(1)
                # env.render()

                state_next, reward, done, info = env.step(action)
                ep_total_reward += reward

                sample_data.append([state_norm, None, act_val_nlogp, self.reward_norm(reward), done])

                # print('reward, done')
                # print(reward, done)

                state = state_next

                if done:
                    state = env.reset()
                    break

            self.ep_total_reward_list.append(ep_total_reward)

            if len(sample_data) > sample_length:
                break

        return sample_data

    def reward_norm(self, reward):

        reward = np.array([reward])
        assert reward.shape == (1,)

        self.rms['reward'].update(reward)

        reward -= self.rms['reward'].mean
        reward /= np.sqrt(self.rms['reward'].var)

        assert np.all(np.isfinite(reward))

        reward = reward[0]
        assert reward.shape == ()

        return reward

def main():
    sampler = Sampler(obs_norm={
        'mean': [0.] * 3,
        'std': [1., 1., 8.],
    })
    sample_data = sampler.run_env(sample_length=1000)

if __name__ == '__main__':
    main()