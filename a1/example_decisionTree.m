% Load X and y variable
load citiesSmall.mat
[n,d] = size(X);

%% Fit decision tree and compute error
depth = 2;
model = decisionTree(X,y,depth);

% Evaluate training error
yhat = model.predict(model,X);
error = sum(yhat ~= y)/n;
fprintf('Error with depth-%d decision tree: %.2f\n',depth,error);

% Plot classifier
figure(1);
classifier2Dplot(X,y,model);
