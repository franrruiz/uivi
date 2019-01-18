function llh = compute_llh_logreg(X, Y, T, vardist)

dim_z = length(vardist.net{1}.b);

logpx = zeros(T,1);
for t=1:T
    % Sample regression coefficients from q
    if strcmp(vardist.peps.pdf, 'standard_normal')
        Eps0 = randn(1, vardist.peps.dim_noise);
    elseif strcmp(vardist.peps.pdf,'uniform')
        Eps0 = rand(1, vardist.peps.dim_noise);
    end
    net = netforward(vardist.net, Eps0);
    Tr_epsilon = net{1}.Z; 
    z = Tr_epsilon  + vardist.sigma.*randn(1,dim_z); 
    
    % 
    logpx(t) = sum( logsigmoid((2*Y-1).*(X*z')) );
end
llh = logsumexp(logpx,1) - log(T);
llh = llh/size(X,1);
