import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
# from plotting import newfig
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


se = 42
np.random.seed(se)
tf.set_random_seed(se)
Tmax = 4
ah = -1/2
bh = 1/3
nlow = (5000, 1)
nT = (5000, 1)
nD = (20000, 1)
lam12 = 2
lam21 = 1
def payoff(strike, St):
    diff = strike-St
    payoff = np.max(np.hstack([np.zeros_like(diff), diff]), axis = 1)
    return payoff.reshape(payoff.shape[0], 1)

s = np.random.uniform(30, 110, nD)
sigma1 = np.random.uniform(0.1, 0.3, nD)
sigma2 = np.random.uniform(sigma1, 0.4, nD)
r = np.random.uniform(0.015, 0.025, nD)
sbar = np.mean(s)
sy = np.std(s)
a_s = -sbar/sy
b_s = 1/sy
stilde = a_s + b_s*s
T = np.random.uniform(0, Tmax, nD[0]).reshape(nD)
t = np.random.uniform(0, T, nD)
ttilde = ah+bh*t
Ttilde = ah+bh*T

print(a_s, b_s)
################# Generating set of auxilliary points for physics regularisation #################

aux = np.hstack([ttilde, stilde, r, sigma1, sigma2, Ttilde])
aux1 = np.hstack([t, s, r, sigma1, sigma2, T])

##################################################################################################

T = np.random.uniform(0, Tmax, nT[0]).reshape(nT)
TtildeT = ah+bh*T
sT = np.random.uniform(30, 110, nT)
stildeT = a_s + b_s*sT
rT = np.random.uniform(0.015, 0.025, nT)
sigma1T = np.random.uniform(0.1, 0.3, nT)
sigma2T = np.random.uniform(sigma1T, 0.40, nT)


################# Generating the set of terminal boundary points #################

ST = np.hstack([TtildeT, stildeT, rT, sigma1T, sigma2T, TtildeT])

################# Payoff function for the Terminal boundary ######################

HT = payoff(70, (stildeT-a_s)/b_s)

##################################################################################

sl = a_s*np.ones_like(sT)
rl = np.random.uniform(0.015, 0.025, nlow)
sigma1l = np.random.uniform(0.1, 0.3, nlow)
sigma2l = np.random.uniform(sigma1l, 0.4, nlow)
Tl = np.random.uniform(0, Tmax, nlow[0]).reshape(nlow)
tl = np.random.uniform(0, T, nlow)
ttildel = ah + bh*tl
Ttildel = ah + bh*Tl

################# Generating the set of terminal boundary points #################

Sl = np.hstack([ttildel, sl, rl, sigma1l, sigma2l, Ttildel])

################# Payoff function for the lower boundary ######################

Hl = np.exp(-rl*(Ttildel-ttildel)/bh)*payoff(70, np.zeros_like(sl))

##################################################################################
S = np.vstack([Sl, ST])
H = np.vstack([Hl, HT])
H = np.hstack([H, H])
input('Press enter to continue>>>>')
pd.DataFrame(aux).to_csv('//home//naman//Finance//auxilliarypoints.csv')
pd.DataFrame(aux1).to_csv('//home//naman//Finance//auxpoints.csv')
pd.DataFrame(S).to_csv('//home//naman//Finance//LowerandTerminalpoints.csv')
pd.DataFrame(H).to_csv('//home//naman//Finance//LowerandTerminalPayoffs.csv')


class PIRegimeSwitch:
    def __init__(self, inputs, outputs, aux, layers, epochs):
        self.epochs = epochs
        self.V1 = outputs[:, 0:1]
        self.V2 = outputs[:, 1:2]
        self.layers = layers
        
        self.t = inputs[:, 0:1] 
        self.s = inputs[:, 1:2]
        self.r = inputs[:, 2:3]
        self.sigma1 = inputs[:, 3:4]
        self.sigma2 = inputs[:, 4:5]
        self.T = inputs[:, 5:6]
        
        self.At = aux[:, 0:1] 
        self.As = aux[:, 1:2]
        self.Ar = aux[:, 2:3]
        self.Asigma1 = aux[:, 3:4]
        self.Asigma2 = aux[:, 4:5]
        self.AT = aux[:, 5:6]
        
        
        self.weights, self.biases = self.initialise(layers)
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self.tp = tf.placeholder(tf.float32, shape = [None, self.t.shape[1]])
        self.sp = tf.placeholder(tf.float32, shape = [None, self.s.shape[1]])
        self.rp = tf.placeholder(tf.float32, shape = [None, self.r.shape[1]])
        self.sigma1p = tf.placeholder(tf.float32, shape = [None, self.sigma1.shape[1]])
        self.sigma2p = tf.placeholder(tf.float32, shape = [None, self.sigma2.shape[1]])
        self.Tp = tf.placeholder(tf.float32, shape = [None, self.T.shape[1]])
        
        self.Atp = tf.placeholder(tf.float32, shape = [None, self.At.shape[1]])
        self.Asp = tf.placeholder(tf.float32, shape = [None, self.As.shape[1]])
        self.Arp = tf.placeholder(tf.float32, shape = [None, self.Ar.shape[1]])
        self.Asigma1p = tf.placeholder(tf.float32, shape = [None, self.Asigma1.shape[1]])
        self.Asigma2p = tf.placeholder(tf.float32, shape = [None, self.Asigma2.shape[1]])        
        self.ATp = tf.placeholder(tf.float32, shape = [None, self.AT.shape[1]])
        
        self.Y = self.approximation(self.tp, self.sp, self.rp, self.sigma1p, self.sigma2p, self.Tp)
        self.hatV1 = self.Y[:, 0:1]
        self.hatV2 = self.Y[:, 1:2]
        self.pde_res = self.pde_residual(self.Atp, self.Asp, self.Arp, self.Asigma1p, self.Asigma2p, self.ATp)
        self.f1 = self.pde_res[0]
        self.f2 = self.pde_res[1]
        
        self.loss1 = tf.reduce_mean(tf.square(self.hatV1-self.V1)) + tf.reduce_mean(tf.square(self.hatV2-self.V2))
        self.loss2 = tf.reduce_mean(tf.square(self.f1)) + tf.reduce_mean(tf.square(self.f2))
        self.loss = self.loss1 + self.loss2
        
        self.optimiser1 = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        self.optimiser2 = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': self.epochs,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    
    def initialise(self, layers):
        weights = []
        biases = []
        L = len(layers)
        n0 = layers[0]
        for l in range(L-1):
            m = layers[l]
            n = layers[l + 1]
            if l >= 1 and l <= L-3:
                sigma = np.sqrt(2 / (m+n0+n))
                W = tf.Variable(tf.truncated_normal([m+n0, n], mean = 0, stddev = sigma, seed = se), dtype = tf.float32)
                b = tf.Variable(tf.zeros([1, n]), dtype = tf.float32)
                weights.append(W)
                biases.append(b)
            else:
                sigma = np.sqrt(2 / (m + n))
                W = tf.Variable(tf.truncated_normal([m, n], mean = 0, stddev=sigma, seed=se), dtype = tf.float32)
                b = tf.Variable(tf.zeros([1, n]), dtype=tf.float32)
                weights.append(W)
                biases.append(b)
        return weights, biases

    def neural_network(self, X, weights, biases):

        L = len(weights) + 1
        A0 = X#2*((X-self.lb)/(self.ub-self.lb))-1
        W = weights[0]
        b = biases[0]
        Z = tf.add(tf.matmul(A0, W), b)
        A = tf.tanh(Z)
        for l in range(1, L-2):
            W = weights[l]
            b = biases[l]
            A1 = tf.concat([A, A0], axis = 1)
            Z = tf.add(tf.matmul(A1, W), b)
            A = tf.tanh(Z)+A
        W = weights[-1]
        b = biases[-1]
        Z = tf.add(tf.matmul(A, W), b)
        Y = Z
        return Y
    def approximation(self, t, s, r, sigma1, sigma2, T):
        u = self.neural_network(tf.concat([t, s, r, sigma1, sigma2, T], axis=1), self.weights, self.biases)
        return u
    
    
    def pde_residual(self, t, s, r, sigma1, sigma2, T):
        NN = self.approximation(t, s, r, sigma1, sigma2, T)
        V1 = NN[:, 0:1]
        V2 = NN[:, 1:2]
        dV1t = tf.gradients(V1, t)[0]
        dV1s = tf.gradients(V1, s)[0]
        dV1ss = tf.gradients(dV1s, s)[0]
        dV2t = tf.gradients(V2, t)[0]
        dV2s = tf.gradients(V2, s)[0]
        dV2ss = tf.gradients(dV2s, s)[0]
        f1 = bh*dV1t - r*V1 + 0.5*sigma1**2*(s-a_s)**2*dV1ss + r*(s-a_s)*dV1s - lam12*(V1-V2)
        f2 = bh*dV2t - r*V2 + 0.5*sigma2**2*(s-a_s)**2*dV2ss + r*(s-a_s)*dV2s - lam21*(V2-V1)
        return [f1, f2]

    def callback(self, loss):
        print('Loss:', loss)

    def train(self):
        feed_dict = {self.tp: self.t, self.sp: self.s, self.rp: self.r, self.sigma1p: self.sigma1, self.sigma2p: self.sigma2, self.Tp: self.T,
                         self.Atp: self.At, self.Asp: self.As, self.Arp: self.Ar, self.Asigma1p: self.Asigma1, self.Asigma2p: self.Asigma2, self.ATp: self.AT}
        self.optimiser2.minimize(self.sess,
                                feed_dict = feed_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

    def train_adam(self, lr, epochs):
        for epoch in range(epochs):
            _, epoch_loss, epoch_loss1, epoch_loss2 = self.sess.run([self.optimiser1, self.loss, self.loss1, self.loss2],
            feed_dict = {self.tp: self.t, self.sp: self.s, self.rp: self.r, self.sigma1p: self.sigma1, self.sigma2p: self.sigma2, self.Tp: self.T,
                         self.Atp: self.At, self.Asp: self.As, self.Arp: self.Ar, self.Asigma1p: self.Asigma1, self.Asigma2p: self.Asigma2, self.ATp: self.AT, self.learning_rate: lr})
            print("Epoch:", epoch+1, "Total Loss:", epoch_loss, "Loss1:", epoch_loss1, "Loss2:", epoch_loss2)

    def predict(self, test_ds):
        Y_star = self.sess.run(self.Y, feed_dict={
            self.tp: test_ds[:, 0:1],
            self.sp: test_ds[:, 1:2],
            self.rp: test_ds[:, 2:3],
            self.sigma1p: test_ds[:, 3:4],
            self.sigma2p: test_ds[:, 4:5],
            self.Tp: test_ds[:, 5:6]
        })
        
        f_star = self.sess.run(self.pde_res, feed_dict={
            self.Atp: aux[:, 0:1],
            self.Asp: aux[:, 1:2],
            self.Arp: aux[:, 2:3],
            self.Asigma1p: aux[:, 3:4],
            self.Asigma2p: aux[:, 4:5],
            self.ATp: aux[:, 5:6]
        })
        return Y_star, f_star
    
    def save_weights_and_biases(self, filepath):
        weights_values = self.sess.run(self.weights)
        biases_values = self.sess.run(self.biases)
        
        # Flatten weights and biases and concatenate them
        weights_flat = [w.flatten().T for w in weights_values]
        biases_flat = [b.flatten().T for b in biases_values]
        
        l = max([len(item) for item in weights_flat])
        
        # Combine all arrays into one
        df = pd.DataFrame([np.ones_like(weights_flat[1])]+weights_flat+biases_flat)

        
        # Save to CSV without header and index
        df.to_csv(filepath, header=False, index=False)
        
layers = [6, 32, 32, 32, 32, 32, 32, 32, 32, 2]#6-48: 1542s, 6-32: 1154s, 8-48: 2070s, 8-32: 1574s, 8-16: 1115s
epochs = 15000
model = PIRegimeSwitch(S, H, aux, layers, epochs)
init = time.time()
model.train()
print('Time Elapsed:', time.time()-init)
print(layers, layers[1])
model.save_weights_and_biases('//home//naman/Finance//WB'+str(len(layers)-2)+'L.csv')
V, _ = model.predict(S)
pd.DataFrame(V).to_csv('//home//naman/Finance//Pred.csv')