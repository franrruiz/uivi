function mean_px = compute_llh_vae(T, S, pxz, vardist, data)
% Compute an importance sampling estimate of the log-evidence on test
%  T: number of samples for the log q(z) term
%  S: number of samples for the importance sampling approximation
%  vardist: variational distribution
%  data: struct containing the data

if(vardist.peps.dim_noise==0)
    T = 1;
end

px = zeros(data.test.N,1);
pxz.inargs{1} = pxz.vae;
for ns=1:data.test.N
    % Sample epsilon and pass through the NN
    Epsilons_basis = randn(T, vardist.peps.dim_noise);
    XX = repmat(data.test.X(ns,:), T, 1);
    net = netforward(vardist.net, [XX, Epsilons_basis]);
    Tr_epsilon_all = net{1}.Z;
    
    % Sample z
    eps = randn(S, vardist.peps.dim_noise);
    XX2 = repmat(data.test.X(ns,:), S, 1);
    net2 = netforward(vardist.net, [XX2, eps]);
    dim_z = size(net2{1}.Z,2);
    z = net2{1}.Z  + bsxfun(@times, vardist.sigma, randn(S, dim_z));
    
    % Approximate log q(z)
    d1 = bsxfun(@rdivide, Tr_epsilon_all, vardist.sigma);
    d2 = bsxfun(@rdivide, z, vardist.sigma);
    diffs2 = bsxfun(@plus, bsxfun(@plus, -2*d1*d2', sum(d1.*d1,2)), sum(d2.*d2,2)');
    logq = - 0.5*dim_z*log(2*pi) - sum(log(vardist.sigma)) - 0.5*diffs2;
    logq = logsumexp(logq,1)' - log(T);
    
    % Evaluate the log-joint
    pxz.inargs{1}{1}.outData = XX2;
    logjoint = pxz.logdensity(z, pxz.inargs{:});
    
    % Importance sampling term
    px(ns) = logsumexp(logjoint - logq, 1) - log(S);
end
mean_px = mean(px);
