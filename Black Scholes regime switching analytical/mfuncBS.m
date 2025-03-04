function m = mfuncBS(tau, S, r, phi, sigma1, sigma2, lam12, lam21, X)
    M = [-lam12 * tau + 0.5 * sigma1^2 * (-1i * phi - phi^2) * tau, lam21 * tau; lam12 * tau, -lam21 * tau + 0.5 * sigma2^2 * (-1i * phi - phi^2) * tau];
    eMX = expm(M) * X';
    eMXI = eMX(1) + eMX(2);
    m = eMXI * exp(1i * phi * log(S) + 1i * phi * r * tau);
end