
% Clear variables and close figures
clear all
close all

% Load data
load basisData.mat % Loads X and y
[n,d] = size(X);

% Split training data into a training and a validation set
ordering = randperm(n);
X = X(ordering,:);
y = y(ordering,:);
Xtrain = X(n/2,:);
ytrain = y(n/2,:);
Xvalid = X(n/2+1:end,:);
yvalid = y(n/2+1:end,:);

% Find best value of RBF kernel parameter,
%   training on the train set and validating on the validation set
minErr = inf;
for sigma = 2.^[-15:15]

    % Train on the training set
    model = leastSquaresRBF(Xtrain,ytrain,sigma);

    % Compute the error on the validation set
    yhat = model.predict(model,Xvalid);
    validError = sum((yhat - yvalid).^2)/(n/2);
    fprintf('Error with sigma = %.3e = %.2f\n',sigma,validError);

    % Keep track of the lowest validation error
    if validError < minErr
        minErr = validError;
        bestSigma = sigma;
    end
end
fprintf('Value of sigma that achieved the lowest validation error was %.3e\n',bestSigma);

% Now fit the model based on the full dataset.
fprintf('Refitting on full training set...\n');
model = leastSquaresRBF(X,y,bestSigma);

% Compute training error
yhat = model.predict(model,X);
trainError = sum((yhat - y).^2)/n;
fprintf('Training error = %.2f\n',trainError);

% Finally, report the error on the test set
t = size(Xtest,1);
yhat = model.predict(model,Xtest);
testError = sum((yhat - ytest).^2)/t;
fprintf('Test error = %.2f\n',testError);

% Plot model
figure(1);
plot(X,y,'b.');
title('Training Data');
hold on
Xhat = [min(X):.1:max(X)]'; % Choose points to evaluate the function
yhat = model.predict(model,Xhat);
plot(Xhat,yhat,'g');
ylim([-300 400]);
