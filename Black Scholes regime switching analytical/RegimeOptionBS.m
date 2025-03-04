clc
clear all
% Main script
rng(42);
n = 25000;
s = unifrnd(40, 100, [n, 1]);
sigma1 = unifrnd(0.1, 0.3, [n, 1]);
% sigma2 = sigma1;
sigma2 = unifrnd(sigma1, 0.4, [n, 1]);
r = unifrnd(0.015, 0.025, [n, 1]);
Tmax = 4;
T = unifrnd(1/12, Tmax, [n, 1]);
t = unifrnd(0, T-1/12, [n, 1]);
start_time = tic;
tau = T - t;
K = 70;
puts = zeros(n, 2);
put = zeros(n, 1);
for i = 1:n
    item = [tau(i), s(i), r(i), sigma1(i), sigma2(i)];
    putprice = RegimeOption(item(1), item(2), item(3), item(4), item(5), K);
    %put(i, 1) = blackScholes(s(i), K, r(i), tau(i), sigma1(i));
    puts(i,1) = putprice(1);
    puts(i,2) = putprice(2);
    fprintf('%d, %0.4f, %0.4f\n', i, puts(i,1), puts(i, 2))
end

total_time = toc(start_time);
fprintf('\nDone in %0.2f minutes.......\n', total_time / 60);

Samples = [t, s, r, sigma1, sigma2, T];
putsblack = puts;
csvwrite('Analytical Samples.csv', Samples);
csvwrite('Analytical Options.csv', putsblack);
