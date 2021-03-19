import tensorflow as tf
import numpy as np
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

from .policy import Policy
from .value_net import value_nn
from .sampler import Sampler

class Agent():
    def __init__(self):

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        self.mblossvals = []

        self.sampler = Sampler(obs_norm={
            'mean': [0.] * 3,
            'std': [1., 1., 8.],
        })

        self.build_network()

        self.build_train()

        # baselines.common.tf_util
        self.sess = get_session()
        initialize()

    def build_network(self):

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):

            obs_dim = 3
            self.policy = Policy(obs_dim=obs_dim)

            self.param_act, self.param_nlogp, self.param_ent, self.param_pd = self.policy.action_dist

            self.value = value_nn(self.policy.obs_ph, obs_dim=obs_dim)

    def build_train(self):

        vf_coef = 0.5
        ent_coef = 0.
        max_grad_norm = 0.5

        # CREATE THE PLACEHOLDERS
        self.A = A = self.get_action_ph()
        self.ADV = ADV = tf.placeholder(tf.float32, [None], name='ADV')
        self.R = R = tf.placeholder(tf.float32, [None], name='Return')
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name='OLDNEGLOGPAC')
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None], name='OLDVPRED')
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = self.get_neglogpac(A, self.param_pd)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(self.param_ent)

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = self.value
        vpredclipped = OLDVPRED + tf.clip_by_value(self.value - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = - ADV * ratio

        pg_losses2 = - ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        # 2. Build our trainer
        self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        # self.save = functools.partial(save_variables, sess=sess)
        # self.load = functools.partial(load_variables, sess=sess)

    def step_train(self):

        nsteps = 512
        nbatch_train = 64

        # Get minibatch
        sample_data = self.sampler.run_env(sample_length=nsteps)
        assert len(sample_data) > nsteps

        # print('sample_data[0]')
        # print(sample_data[0])
        # assert 0

        array = np.array
        float32 = np.float32
        
        [array([ 0.10714442,  0.99424347, -0.06181507]), 

        None, 

        {'act': [array([[0.5577679]], dtype=float32)], 'value': array([-0.15849566], dtype=float32), 'neglogpacs': [array([0.99973834], dtype=float32)]}, 

        -2.8712233299991724e-05, 

        False]

        mb_obs = [i[0] for i in sample_data[:nsteps]]
        mb_actions = [i[2]['act'][0][0] for i in sample_data[:nsteps]]
        mb_neglogpacs = [i[2]['neglogpacs'][0][0] for i in sample_data[:nsteps]]

        # GAE
        # get mb_advs and mb_returns

        mb_rewards = [i[-2] for i in sample_data[:nsteps]]
        mb_dones = [i[-1] for i in sample_data[:nsteps]]
        mb_values = [i[2]['value'][0] for i in sample_data[:nsteps]]

        lam = 0.95
        gamma = 0.98

        last_values = sample_data[nsteps][2]['value'][0]

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextvalues = last_values
            else:
                nextvalues = mb_values[t+1]
            nextnonterminal = 1.0 - mb_dones[t]
            delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        # shuffle & mini sample
        self.shuffle_mini_batch(mb_obs, mb_returns, mb_actions, mb_values, mb_neglogpacs, nsteps, nbatch_train)

    def shuffle_mini_batch(self, obs, returns, actions, values, neglogpacs, nsteps, nbatch_train):

        # Index of each element of batch_size
        # Create the indices array
        inds = np.arange(nsteps)
        # Randomize the indexes
        np.random.shuffle(inds)
        # 0 to batch_size with batch_train_size step
        for start in range(0, nsteps, nbatch_train):
            end = start + nbatch_train
            mbinds = inds[start:end]
            slices = ([arr[i] for i in mbinds] for arr in (obs, returns, actions, values, neglogpacs))
            self.mblossvals.append(self.train(*slices))


    def train(self, obs, returns, actions, values, neglogpacs):

        cliprange = 0.2
        lr = 1e-4

        returns = np.array(returns)
        values = np.array(values)

        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.policy.obs_ph : obs,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }

        # action feed
        td_map.update({self.A: actions})

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]

    def get_action_ph(self):

        return tf.placeholder(tf.float32, [None, 1], name="param_act")

    def get_neglogpac(self, param_act, param_pd):

        return param_pd.neglogp(param_act)

def main():

    print('')
    print('')
    from matplotlib import pyplot as plt

    agent = Agent()
    for _ in range(10):
        agent.step_train()

    plt.plot(agent.sampler.ep_total_reward_list)
    plt.show()

if __name__ == '__main__':
    main()