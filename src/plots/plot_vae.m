%% Reconstructed data  
epsilon = randn(data.N, vardist.peps.dim_noise);   
net = netforward(vardist.net, [data.X, epsilon]);
Tr_epsilon = net{1}.Z; 
z = Tr_epsilon + bsxfun(@times, vardist.sigma, randn(data.N, dim_z)); 
netReco = netforward(pxz.vae, z); 

% Plot reconstructed data 
S = 10; 
figure;
cnt = 0; 
for i=1:S  
    cnt = cnt + 1;
    subtightplot(2,S,cnt);
    if(strcmp(pxz.dataName, 'bmnist'))
        imagesc(reshape(data.X(i,:),28,28)');
    elseif(strcmp(pxz.dataName, 'fashionmnist'))
        imagesc(reshape(data.X(i,:),28,28));
    end
    axis off;
    axis square;
    colormap('gray'); 
    subtightplot(2,S,S+cnt);     
    if(strcmp(pxz.dataName, 'bmnist'))
        imagesc(reshape(netReco{1}.Z(i,:), 28,28)');
    elseif(strcmp(pxz.dataName, 'fashionmnist'))
        imagesc(reshape(netReco{1}.Z(i,:), 28,28));
    end
    axis off;
    axis square;
    colormap('gray'); 
end
box off;
name = [param.outdir 'plot_reconstruct_vae_' pxz.dataName '_' param.method];
figurepdf(6,1);
print('-dpdf', [name '.pdf']);
