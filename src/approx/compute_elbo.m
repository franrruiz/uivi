function elbo = compute_elbo(T, S, pxz, vardist)

if(vardist.peps.dim_noise==0)
    T = 1;
end

dim_z = length(vardist.net{1}.b);

if strcmp(vardist.peps.pdf,'standard_normal')
    Eps0 = randn(T, vardist.peps.dim_noise);
elseif strcmp(vardist.peps.pdf,'uniform')
    Eps0 = rand(T, vardist.peps.dim_noise);
end
net = netforward(vardist.net, Eps0);
Tr_epsilon_all = net{1}.Z;

const = - log(T) - 0.5*dim_z*log(2*pi) - sum(log(vardist.sigma));
elbo = 0;
for s=1:S
    if strcmp(vardist.peps.pdf,'standard_normal')
        eps = randn(1, vardist.peps.dim_noise);
    elseif strcmp(vardist.peps.pdf,'uniform')
        eps = rand(1, vardist.peps.dim_noise);
    end
    net = netforward(vardist.net, eps);
    z = net{1}.Z  + vardist.sigma.*randn(1,dim_z);
    logpxz = pxz.logdensity(z, pxz.inargs{:});
    diffs = bsxfun(@minus, z, Tr_epsilon_all);
    diffs = bsxfun(@times, diffs, 1./vardist.sigma);
    diffs2 = - 0.5*sum(diffs.^2, 2);
    entr = - const - logsumexp(diffs2, 1);
    elbo = elbo + logpxz + entr;
end
elbo = elbo/S; 
