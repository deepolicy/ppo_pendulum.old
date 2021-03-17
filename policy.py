import tensorflow as tf
import numpy as np

from .distributions import CategoricalPd, DiagGaussianPd

def get_pd_discrete(latent):
    pd = CategoricalPd(latent)
    act = pd.sample()
    return act, pd.neglogp(act), pd.entropy(), pd

def get_pd_continuous(mean, i_size=None, logstd=None, logstd_name='logstd'):
    if logstd == None:
        logstd = tf.get_variable(name=logstd_name, shape=[1, i_size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=-1)
    else:
        pdparam = tf.concat([mean, logstd], axis=-1)
    diag_gaussian_pd = DiagGaussianPd(pdparam)
    act = diag_gaussian_pd.sample()
    return act, diag_gaussian_pd.neglogp(act), diag_gaussian_pd.entropy(), diag_gaussian_pd, logstd

class Policy(object):
    def __init__(self, obs_dim):
    
        self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))

        param_act, param_nlogp, param_ent, param_pd = self.forward(self.obs_ph)
        self.action_dist = [param_act, param_nlogp, param_ent, param_pd]


    def forward(self, state, reuse=False):

        act_dim = 1

        with tf.variable_scope('policy'):

            ly1 = tf.layers.dense(state, 64, activation=tf.nn.relu, name='ly1')
            ly2 = tf.layers.dense(ly1, act_dim, activation=None, name='ly2')
            action_nodes = ly2

            # action head
            mean = action_nodes
            i_size = act_dim
            param_act, param_nlogp, param_ent, param_pd, logstd = get_pd_continuous(mean, i_size=i_size, logstd_name='logstd')

        return param_act, param_nlogp, param_ent, param_pd

def main():
    assert 0
    obs_dim = 9
    policy = Policy(obs_dim=obs_dim)

    # Create session
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    # Run sess
    def get_actions(obs):
        fetches = {
            "act": [policy.action_dist[i] for i in [0, 4]],
        }

        feed_dict = {
            policy.obs_ph: obs,
        }

        results = sess.run(fetches, feed_dict)

        print('results')
        print(results)

        '''
        results
        {'act': [array([0, 0, 2, 1, 0, 0, 2, 1, 1, 2]), {'run': array([[-0.25004   ],
               [-0.7016165 ],
               [ 0.53156084],
               [-0.8303973 ],
               [ 0.06244882],
               [ 1.0302209 ],
               [ 0.64292276],
               [-2.4401724 ],
               [-0.31313938],
               [ 0.04702389]], dtype=float32), 'hop': array([[-0.2662964 ],
               [ 1.559813  ],
               [ 0.85801864],
               [ 0.5008656 ],
               [-0.08716473],
               [-1.4858663 ],
               [ 0.44744676],
               [ 0.8131449 ],
               [-0.8181738 ],
               [-0.48000756]], dtype=float32), 'leap': array([[ 0.23802471],
               [ 0.29655567],
               [-0.5099993 ],
               [ 0.37545612],
               [-0.13615723],
               [ 0.3219884 ],
               [ 0.6829447 ],
               [ 0.06417565],
               [ 0.49436846],
               [ 0.67584014]], dtype=float32)}]}
        '''
    batch_size = 10
    obs = np.random.rand(batch_size, obs_dim)

    get_actions(obs)


if __name__ == '__main__':
    main()