import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time
import pandas as pd

states0 = pd.read_csv('States0.csv', header = None).values
states1 = pd.read_csv('States1.csv', header = None).values

def generate_paths(S0, strike, V0, r, k, theta, sigma1, sigma2, rho, T, num_paths, states):
    dt = 1/252
    t = np.linspace(0, T, int(T/dt))
    req_states = sigma2*np.ones_like(states[:num_paths])
    req_states[states[:num_paths] == 0] = sigma1
    N1 = np.random.normal(0, np.sqrt(dt), (num_paths, len(t)))
    Z = np.random.normal(0, np.sqrt(dt), (num_paths, len(t)))
    N2 = rho*N1+np.sqrt(1-rho**2)*Z
    S = np.zeros_like(N1)
    V = np.zeros_like(S)
    S[:, 0:1] = S0
    V[:, 0:1] = V0
    for i in range(len(t)-1):
#         S[:, i+1:i+2] = S[:, i:i+1]+r*S[:, i:i+1]*dt+np.sqrt(V[:, i:i+1])*S[:, i:i+1]*N1[:, i:i+1]
        S[:, i+1:i+2] = S[:, i:i+1]*np.exp((r-0.5*V[:, i:i+1])*dt+np.sqrt(V[:, i:i+1])*N1[:, i:i+1])
        V1 = V[:, i:i+1]+ k*(theta-V[:, i:i+1])*dt+req_states[:, i:i+1]*np.sqrt(V[:, i:i+1])*N2[:, i:i+1]
        V1[V1<0] = 0
        V[:, i+1:i+2] = V1
    return S, V

num_paths = 10
S0 = 45
strike = 70
V0 = 0.05
r = 0.02
k = 2
theta = 0.1
sigma1 = 0.05
sigma2 = 0.9
rho = -0.8
T = 4
start = time.time()
paths = []
puts1 = 0
d = {0: states0, 1:states1}
kk = 1
start = time.time()
for l in range(kk):
    for i in range(2):
        s,v = generate_paths(S0, strike, V0, r, k, theta, sigma1, sigma2, rho, T, num_paths, d[i])
        paths.append(s)
    dt = 1/252
    l = len(paths[0][0])
    puts = [[],[]]
    std = [[], []]
    for i in range(2):
        payoffs = np.maximum(0, strike-paths[i])
        for j in range(0, T*252):
            puts[i].append(np.mean(np.exp(-r*j*dt)*payoffs[:, j:j+1]))
            std[i].append(np.std(np.exp(-r*j*dt)*payoffs[:, j:j+1]))
    puts1 += np.array(puts)
puts1 /= kk
print('Time taken:', time.time()-start)

plt.plot(np.array(paths[0]).T)
plt.show()