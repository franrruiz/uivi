function [block, st, perm] = takeNextBatch(N, numBatch, st, perm)  

% take a minibatch 
if (st+numBatch-1) <= N
    block = perm(st:st+numBatch-1);
    st = st+numBatch;
else
    st = 1; 
    perm = randperm(N);
    block = perm(st:st+numBatch-1);
    st = st+numBatch;
end 