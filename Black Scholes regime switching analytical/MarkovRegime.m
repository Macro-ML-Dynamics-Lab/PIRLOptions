function y = MarkovRegime(num_paths, init_state, lam12, lam21, T)
    Q = [-lam12, lam12; lam21, -lam21];
    dt = 1/252;
    t = linspace(0, T, T*1/dt);
    states = zeros(num_paths,length(t))+init_state;
    for i = 1:num_paths
        for j = 1:length(t)-1
            P = expm(Q*t(j));
            current_state = states(i,j);
            u = rand;
            if current_state == 0
                if u<P(1,2)
                    current_state = 1;
                end
            elseif current_state == 1
                if u<P(2,1)
                    current_state = 0;
                end
            end
            states(i, j+1) = current_state;
        end
    end
    y = states;
