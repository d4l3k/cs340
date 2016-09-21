% Load X and y variable
load newsgroups.mat
[N,D] = size(X);
depth =2 ;
model = decisionTree(X,y, depth);
% Evaluate training error
yhat = model.predictFunc(model,X);
error = sum(yhat ~= y)/N;




