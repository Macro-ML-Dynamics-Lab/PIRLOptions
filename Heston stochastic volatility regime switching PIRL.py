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
nlow = (5000, 1)
nT = (5000, 1)
nD = (30000, 1)
lam12 = 2
lam21 = 1
strike = 70
tscale = 4
ah = -1/2
bh = 1/tscale
Tmax = 3
def payoff(strike, St):
    diff = strike*(1-St)
    payoff = np.max(np.hstack([np.zeros_like(diff), diff]), axis = 1)
    return payoff.reshape(payoff.shape[0], 1)

np.random.seed(se)
s = np.random.uniform(40, 100, nD)/strike
v = np.random.uniform(0.01, 0.1, nD)
Y = np.hstack([s,v])
r = np.random.uniform(0.015, 0.025, nD)
k = np.random.uniform(1.4, 2.6, nD)
sigma1 = np.random.uniform(0.1, 0.9, nD)
sigma2 = np.random.uniform(0.1, 0.9, nD)
gamma =  np.random.uniform(0.01, 0.1, nD)
rho = np.random.uniform(-0.85, -0.55, nD)
T = np.random.uniform(0, Tmax, nD)
t = np.random.uniform(0, T)
ttilde = ah+bh*t
Ttilde = ah+bh*T

################# Generating set of auxilliary points for physics regularisation #################

aux = np.hstack([ttilde, Y, r, k, gamma, sigma1, sigma2, rho, Ttilde])

##################################################################################################

T = np.random.uniform(0, Tmax, nT)
sT = np.random.uniform(40, 100, nT)/strike
vT = np.random.uniform(0.01, 0.1, nT)
YT = np.hstack([sT,vT])
rT = np.random.uniform(0.01, 0.025, nT)
kT = np.random.uniform(1.4, 2.6, nT)
sigma1T = np.random.uniform(0.1, 0.9, nT)
sigma2T = np.random.uniform(0.1, 0.9, nT)
gammaT =  np.random.uniform(0.01, 0.1, nT)
rhoT = np.random.uniform(-0.85, -0.55, nT)
TtildeT = ah+bh*T

################# Generating the set of terminal boundary points #################

ST = np.hstack([TtildeT, YT, rT, kT, gammaT, sigma1T, sigma2T, rhoT, TtildeT])

################# Payoff function for the Terminal boundary ######################

HT = payoff(strike, sT)

##################################################################################

vl = np.random.uniform(0.01, 0.1, nlow)
sl = np.zeros_like(vl)
Ybarl = np.hstack([sl, vl])
rl = np.random.uniform(0.015, 0.025, nlow)
kl = np.random.uniform(1.4, 2.6, nlow)
sigma1l = np.random.uniform(0.1, 0.9, nlow)
sigma2l = np.random.uniform(0.1, 0.9, nlow)
gammal =  np.random.uniform(0.01, 0.1, nlow)
rhol = np.random.uniform(-0.85, -0.55, nlow)
Tl = np.random.uniform(0, Tmax, nlow)
tl = np.random.uniform(0, Tl)
ttildel = ah+bh*tl
Ttildel = ah+bh*Tl

################# Generating the set of lower boundary points ####################

Sl = np.hstack([ttildel, Ybarl, rl, kl, gammal, sigma1l, sigma2l, rhol, Ttildel])

################# Payoff function for the lower boundary #########################

Hl = np.exp(-rl*(Ttildel-ttildel)/bh)*payoff(strike, np.zeros_like(sl))

##################################################################################

S = np.vstack([Sl, ST])
H = np.vstack([Hl, HT])
H = np.hstack([H, H])

pd.DataFrame(aux).to_csv('//home//naman//Finance//auxilliarypoints.csv')
pd.DataFrame(S).to_csv('//home//naman//Finance//LowerandTerminalpoints.csv')
pd.DataFrame(H).to_csv('//home//naman//Finance//LowerandTerminalPayoffs.csv')


class PIHestonRegimeSwitch:
    def __init__(self, inputs, outputs, aux, layers, epochs):
        self.V1 = outputs[:, 0:1]
        self.V2 = outputs[:, 1:2]
        self.layers = layers

        self.t =  inputs[:, 0:1] 
        self.s = inputs[:, 1:2]
        self.v = inputs[:, 2:3]
        self.r = inputs[:, 3:4]
        self.k = inputs[:, 4:5]
        self.gamma = inputs[:, 5:6]
        self.sigma1 = inputs[:, 6:7]
        self.sigma2 = inputs[:, 7:8]
        self.rho = inputs[:, 8:9]
        self.T = inputs[:, 9:10]
      
        self.At =  aux[:, 0:1]
        self.As = aux[:, 1:2]
        self.Av = aux[:, 2:3]
        self.Ar = aux[:, 3:4]
        self.Ak = aux[:, 4:5]
        self.Agamma = aux[:, 5:6]
        self.Asigma1 = aux[:, 6:7]
        self.Asigma2 = aux[:, 7:8]
        self.Arho = aux[:, 8:9]
        self.AT = aux[:, 9:10]

        self.weights, self.biases = self.initialise(layers)
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        
        self.tp = tf.placeholder(tf.float32, shape = [None, self.t.shape[1]])
        self.sp = tf.placeholder(tf.float32, shape = [None, self.s.shape[1]])
        self.vp = tf.placeholder(tf.float32, shape = [None, self.v.shape[1]])
        self.rp = tf.placeholder(tf.float32, shape = [None, self.r.shape[1]])
        self.kp = tf.placeholder(tf.float32, shape = [None, self.k.shape[1]])
        self.gammap = tf.placeholder(tf.float32, shape = [None, self.gamma.shape[1]])
        self.sigma1p = tf.placeholder(tf.float32, shape = [None, self.sigma1.shape[1]])
        self.sigma2p = tf.placeholder(tf.float32, shape = [None, self.sigma2.shape[1]])
        self.rhop = tf.placeholder(tf.float32, shape = [None, self.rho.shape[1]])
        self.Tp = tf.placeholder(tf.float32, shape = [None, self.T.shape[1]])
        
        self.Atp = tf.placeholder(tf.float32, shape = [None, self.At.shape[1]])
        self.Asp = tf.placeholder(tf.float32, shape = [None, self.As.shape[1]])
        self.Avp = tf.placeholder(tf.float32, shape = [None, self.Av.shape[1]])
        self.Arp = tf.placeholder(tf.float32, shape = [None, self.Ar.shape[1]])
        self.Akp = tf.placeholder(tf.float32, shape = [None, self.Ak.shape[1]])
        self.Agammap = tf.placeholder(tf.float32, shape = [None, self.Agamma.shape[1]])
        self.Asigma1p = tf.placeholder(tf.float32, shape = [None, self.Asigma1.shape[1]])
        self.Asigma2p = tf.placeholder(tf.float32, shape = [None, self.Asigma2.shape[1]])
        self.Arhop = tf.placeholder(tf.float32, shape = [None, self.Arho.shape[1]])
        self.ATp = tf.placeholder(tf.float32, shape = [None, self.AT.shape[1]])

        
        self.Y = self.approximation(self.tp, self.sp, self.vp, self.rp, self.kp, self.gammap, self.sigma1p, self.sigma2p, self.rhop, self.Tp)
        self.hatV1 = self.Y[:, 0:1]
        self.hatV2 = self.Y[:, 1:2]
        self.pde_res = self.pde_residual(self.Atp, self.Asp, self.Avp, self.Arp, self.Akp, self.Agammap, self.Asigma1p, self.Asigma2p, self.Arhop, self.ATp)
        self.f1 = self.pde_res[0]
        self.f2 = self.pde_res[1]
        
        self.loss1 = tf.reduce_mean(tf.square(self.hatV1-self.V1)) + tf.reduce_mean(tf.square(self.hatV2-self.V2))
        self.loss2 = tf.reduce_mean(tf.square(self.f1)) + tf.reduce_mean(tf.square(self.f2))
        self.loss = self.loss1 + self.loss2
        
        self.optimiser = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 15000,
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
        A0 = X
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

    def approximation(self, t, s, v, r, k, gamma, sigma1, sigma2, rho, T):
        u = self.neural_network(tf.concat([t, s, v, r, k, gamma, sigma1, sigma2, rho, T], axis=1), self.weights, self.biases)
        return u
    
    def pde_residual(self, t, s, v, r, k, gamma, sigma1, sigma2, rho, T):
        NN = self.approximation(t, s, v, r, k, gamma, sigma1, sigma2, rho, T)
        V1 = NN[:, 0:1]
        V2 = NN[:, 1:2]
        dV1t = tf.gradients(V1, t)[0]
        dV1s = tf.gradients(V1, s)[0]
        dV1ss = tf.gradients(dV1s, s)[0]
        dV1v = tf.gradients(V1, v)[0]
        dV1sv = tf.gradients(dV1v, s)[0]
        dV1vv = tf.gradients(dV1v, v)[0]
        dV2t = tf.gradients(V2, t)[0]
        dV2s = tf.gradients(V2, s)[0]
        dV2ss = tf.gradients(dV2s, s)[0]
        dV2v = tf.gradients(V2, v)[0]
        dV2sv = tf.gradients(dV2v, s)[0]
        dV2vv = tf.gradients(dV2v, v)[0]
        f1 = bh*dV1t-r*V1+r*s*dV1s+k*(gamma-v)*dV1v+1/2*s**2*v*dV1ss+rho*sigma1*s*v*dV1sv+1/2*sigma1**2*v*dV1vv-lam12*(V1-V2)
        f2 = bh*dV2t-r*V2+r*s*dV2s+k*(gamma-v)*dV2v+1/2*s**2*v*dV2ss+rho*sigma1*s*v*dV2sv+1/2*sigma1**2*v*dV2vv-lam21*(V2-V1)
        return [f1, f2]

    def callback(self, loss):
        print('Loss:', loss)

    def train(self):
        feed_dict = {self.tp: self.t, self.sp: self.s, self.vp: self.v, self.rp: self.r, self.kp: self.k,
                         self.gammap: self.gamma, self.sigma1p: self.sigma1, self.sigma2p: self.sigma2, self.rhop: self.rho, self.Tp: self.T,
                         self.Atp: self.At, self.Asp: self.As, self.Avp: self.Av, self.Arp: self.Ar, self.Akp: self.Ak,
                         self.Agammap: self.Agamma, self.Asigma1p: self.Asigma1, self.Asigma2p: self.Asigma2, self.Arhop: self.Arho, self.ATp: self.AT}
        self.optimiser.minimize(self.sess,
                                feed_dict = feed_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)


    def predict(self, test_ds):
        Y_star = self.sess.run(self.Y, feed_dict={
            self.tp: test_ds[:, 0:1],
            self.sp: test_ds[:, 1:2],
            self.vp: test_ds[:, 2:3],
            self.rp: test_ds[:, 3:4],
            self.kp: test_ds[:, 4:5],
            self.gammap: test_ds[:, 5:6],
            self.sigma1p: test_ds[:, 6:7],
            self.sigma2p: test_ds[:, 7:8],
            self.rhop: test_ds[:, 8:9],
            self.Tp: test_ds[:, 9:10]
        })
        
        f_star = self.sess.run(self.pde_res, feed_dict={
            self.Atp: aux[:, 0:1],
            self.Asp: aux[:, 1:2],
            self.Avp: aux[:, 2:3],
            self.Arp: aux[:, 3:4],
            self.Akp: aux[:, 4:5],
            self.Agammap: aux[:, 5:6],
            self.Asigma1p: aux[:, 6:7],
            self.Asigma2p: aux[:, 7:8],
            self.Arhop: aux[:, 8:9],
            self.ATp: aux[:, 9:10]
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
        
layers = [10, 48, 48, 48, 48, 48, 48, 2]
epochs = None
print('Hello')
model = PIHestonRegimeSwitch(S, H, aux, layers, epochs)
init = time.time()
model.train()
print('Time Elapsed:', time.time()-init)
model.save_weights_and_biases('//home//naman/Finance//WB.csv')
V, _ = model.predict(S)
pd.DataFrame(V).to_csv('//home//naman/Finance//Pred.csv')