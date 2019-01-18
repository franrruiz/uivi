function mean_px = compute_llh_vae_explicit(S, pxz, vardist, data)
% Compute an importance sampling estimate of the log-evidence on test
%  S: number of samples for the importance sampling approximation
%  vardist: variational distribution
%  data: struct containing the data

px = zeros(data.test.N,1);
pxz.inargs{1} = pxz.vae;
for ns=1:data.test.N
    % Sample z
    XX = data.test.X(ns,:);
    netMu = netforward(vardist.netMu, XX);
    netSigma = netforward(vardist.netSigma, XX);
    dim_z = size(netMu{1}.Z,2);
    eta = randn(S, dim_z);
    z = bsxfun(@plus, netMu{1}.Z, bsxfun(@times, netSigma{1}.Z, eta));
    
    % Gaussian log q(z)
    logq = -0.5*dim_z*log(2*pi) - sum(log(netSigma{1}.Z)) - 0.5*sum(eta.^2, 2);
    
    % Evaluate the log-joint
    pxz.inargs{1}{1}.outData = repmat(XX, [S 1]);
    logjoint = pxz.logdensity(z, pxz.inargs{:});
    
    % Importance sampling term
    px(ns) = logsumexp(logjoint - logq, 1) - log(S);
end
mean_px = mean(px);
