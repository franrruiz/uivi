function [out, gradz] = logdensityGaussianMixture(z, mu, L, w) 
% z  : (1 x n) latent variable
% mu : cell of (1 x n)  mean vectors
% L  : cell of lower triangular Cholesky decomposition of the 
%      covariance matrices
% w  : vector (1 x c) of weights


logpdf = zeros(1, length(mu));
logw = log(w);
for cc=1:length(mu)
    logpdf(cc) = logpdf_gaussian(z,mu{cc},L{cc});
end
out = logsumexp(logpdf + logw, 2);


if nargout > 1
    qc_z = softmax(logpdf + logw, 2);
    
    dlogqc_dz = zeros(length(z), length(mu));
    for cc=1:length(mu)
        dlogqc_dz(:,cc) = -(L{cc}')\(L{cc}\(z-mu{cc})');
    end
    
    gradz = sum(bsxfun(@times, dlogqc_dz,  qc_z), 2)';
end
