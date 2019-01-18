randn('seed',1);
rand('seed',1);

my_color = [0.466 0.674 0.188;   % green
            0 0.447 0.741;       % blue
            0.929 0.694 0.125;   % orange
            0.85 0.325 0.098;    % red
            0.494 0.184 0.556];  % purple

% Obtain approximate samples from the variational distribution 
Eps0 = zeros(T,param.dim_noise); 
zAppr = zeros(T,dim_z); 
Tr_epsilon_all = zeros(T,dim_z); 
for t=1:T
   if strcmp(vardist.peps.pdf,'standard_normal')
       Eps0(t,:) = randn(1, vardist.peps.dim_noise);
   elseif strcmp(vardist.peps.pdf,'uniform')
       Eps0(t,:) = rand(1, vardist.peps.dim_noise);
   end
   net = netforward(vardist.net, Eps0(t,:));
   Tr_epsilon = net{1}.Z; 
   zAppr(t,:) = Tr_epsilon  + vardist.sigma.*randn(1,dim_z); 
   Tr_epsilon_all(t,:) = Tr_epsilon;
end

% Obtain contour of the true distribution
figure;
if(strcmp(pxz.model, 'banana'))
    x = linspace(-3,3);
    y = linspace(-8,1);
elseif(strcmp(pxz.model, 'gaussmix'))
    if(strcmp(pxz.data, 'xshape'))
        x = linspace(-6,6);
        y = linspace(-6,6);
    elseif(strcmp(pxz.data, 'multimodal'))
        x = linspace(-6,6);
        y = linspace(-4,4);
    end
end
[X,Y] = meshgrid(x,y);
for i=1:length(x)
    for j=1:length(y)
        Z(i,j) = pxz.logdensity([x(i) y(j)], pxz.inargs{:});
    end
end

% Plot
[cs, h] = contour(X,Y,exp(Z)','Color',my_color(3,:), 'Linewidth',0.8);
box off;
name = [param.outdir pxz.dataName '_' param.method '_ContourSamples'];
hold on;
plot(zAppr(1:10:end,1),zAppr(1:10:end,2),'.','Color',my_color(2,:),'MarkerSize',7);
if(strcmp(pxz.model, 'banana'))
    title('banana');
elseif(strcmp(pxz.model, 'gaussmix'))
    if(strcmp(pxz.data, 'xshape'))
        title('x-shaped');
    elseif(strcmp(pxz.data, 'multimodal'))
        title('multimodal');
    else
        error(['Unknown data name: ' pxz.data]);
    end
else
    error(['Unknown model: ' pxz.model]);
end
figurepdf(3,3);
print('-dpdf', [name '.pdf']);

