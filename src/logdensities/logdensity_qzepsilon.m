function [out, gradepsilon] = logdensity_qzepsilon(epsilon, z, vardist, minibatch) 
% 
%


k = size(epsilon,2);

if nargin == 3 
%    
    net = netforward(vardist.net, epsilon);
    
    if isfield(vardist, 'netVar')
       netVar = netforward(vardist.netVar, epsilon);
       vardist.sigma = exp( netVar{1}.Z );
    end
%    
elseif nargin > 3 
%    
    % useful when we run a variational autoendocer and we concatenate the
    % minibatch inputs with the noises epsilon
    net = netforward(vardist.net, [minibatch, epsilon]);
    if isfield(vardist, 'netVar')
       netVar = netforward(vardist.netVar, [minibatch, epsilon]);
       vardist.sigma = exp(netVar{1}.Z);
    end
%    
end
Tr_epsilon = net{1}.Z; 
diff = bsxfun(@rdivide, z - Tr_epsilon, vardist.sigma); 
diff2 = diff.^2; 
if isfield(vardist, 'netVar')
    out = - sum(netVar{1}.Z, 2) - 0.5*sum( diff2, 2) - 0.5*sum(epsilon.*epsilon, 2); 
else
    out = - 0.5*sum( diff2, 2) - 0.5*sum(epsilon.*epsilon, 2); 
end


if nargout > 1
%      
   Deltas = bsxfun(@rdivide, diff, vardist.sigma);
   gradepsilon = netbackpropagationGradofInput(net, Deltas, 1);
  
   gradepsilon = gradepsilon(:, end-k+1:end);
  
   if isfield(vardist, 'netVar')
      gradepsilonVar = netbackpropagationGradofInput(netVar, - ones( size(diff) ) + diff2, 1);
      gradepsilon = gradepsilon + gradepsilonVar(:, end-k+1:end);
   end
   
   gradepsilon = gradepsilon - epsilon; 
%   
end
