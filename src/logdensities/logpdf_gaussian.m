function y=logpdf_gaussian(x,mm,L)

aux = L\((x-mm)');
y = -0.5*length(x)*sqrt(2*pi) - sum(log(diag(L))) - 0.5*(aux'*aux);
