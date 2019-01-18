function llh = compute_llh_softmaxClass(X, Y, T, vardist)

dim_z = length(vardist.net{1}.b);
K = size(Y,2);
D = size(X,2);

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
    Xz = X*reshape(z, K, D)';
    logpx(t) = sum(sum( Y.*Xz )) - sum(logsumexp(Xz, 2));
end
llh = logsumexp(logpx,1) - log(T);
llh = llh/size(X,1);
