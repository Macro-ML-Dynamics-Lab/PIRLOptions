function puts = RegimeOption(tau, S, r, sigma1, sigma2, K)
    lam12 = 2;
    lam21 = 1;
    puts = [];
    states = {[1, 0], [0, 1]};
    for j = 1:2
        X = states{j};
        mbar = @(phi) mfuncBS(tau, S, r, phi-1i, sigma1, sigma2, lam12, lam21, X) / mfuncBS(tau, S, r, -1i, sigma1, sigma2, lam12, lam21, X);
        m = @(phi) mfuncBS(tau, S, r, phi, sigma1, sigma2, lam12, lam21, X);
        ff1 = @(phi) real(exp(-1i * phi * log(K)) ./ (1i * phi) .* mbar(phi));
        ff2 = @(phi) real(exp(-1i * phi * log(K)) ./ (1i * phi) .* m(phi));

        aa = 0.001:0.01:35;
        f1 = arrayfun(ff1, aa);
        f2 = arrayfun(ff2, aa);

        P1 = 0.5 + 1 / pi * trapz(aa, f1);
        P2 = 0.5 + 1 / pi * trapz(aa, f2);

        C1 = exp(-r * tau) * real(m(-1i) * P1 - K * P2);
        U1 = C1 - S + K * exp(-r * tau);
        puts = [puts; U1];
    end
end