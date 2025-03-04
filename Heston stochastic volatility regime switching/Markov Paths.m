clc
clear all
lam12 = 2;
lam21 = 3;
format long
T = 5;
dt = 1/252;
num_paths = 50000;
init_state = 1;
states = MarkovRegime(num_paths, init_state, lam12, lam21, T);
csvwrite('C:\\Users\\HPC\\Documents\\Physics Informed Neural Networks\\Finance\\States1.csv', states)
init_state = 0;
states = MarkovRegime(num_paths, init_state, lam12, lam21, T);
csvwrite('C:\\Users\\HPC\\Documents\\Physics Informed Neural Networks\\Finance\\States0.csv', states)
disp('Done')
% t = linspace(0, T, T*1/dt);
% S0 = 40;
% r = 0.02;
% rho = -0.8;
% strike = 70;
% v0 = 0.05;
% sigma1 = 0.15;
% sigma2 = 0.5;
% k = 2.;
% theta = 0.08;
% prices = real(S0*ones(num_paths, length(t)));
% vols = real(v0*ones(num_paths, length(t)));
% N1 = sqrt(dt)*randn(num_paths, length(t));
% Z = sqrt(dt)*randn(num_paths, length(t));
% N2 = rho*N1+sqrt(1-rho^2)*Z;
% sigmas = [sigma1; sigma2];
% for j = 1:num_paths
%     for i = 1:length(t)-1
%         prices(j, i+1) = real(prices(j,i)+r*prices(j, i)*dt+sqrt(vols(j, i))*prices(j, i)*N1(j, i));
%         vols(j, i+1) = real(vols(j,i)+k*(theta-vols(j, i))*dt+sigmas(states(j,i)+1)*max(0, sqrt(vols(j, i)))*N2(j, i));
%     end
% end
% plot(t, prices)