function [model] = logLinearClassifier(X,y)
% Classification using one-vs-all with logistic loss.

% Compute sizes
[n,d] = size(X);
k = max(y);

W = zeros(d,k); % Each column is a classifier
for c = 1:k
    yc = ones(n,1); % Treat class 'c' as (+1)
    yc(y ~= c) = -1; % Treat other classes as (-1)
    % W(:,c) = (X'*X)\(X'*yc);

    maxFunEvals = 400; % Maximum number of evaluations of objective
    verbose = 1; % Whether or not to display progress of algorithm
    w0 = zeros(d,1);
    W(:,c) = findMin(@logisticLoss,w0,maxFunEvals,verbose,X,yc);
end

model.W = W;
model.predict = @predict;
end

function [yhat] = predict(model,X)
    W = model.W;
    [~,yhat] = max(X*W,[],2);
end

function [f,g] = logisticLoss(w,X,y)
yXw = y.*(X*w);
f = sum(log(1 + exp(-yXw))); % Function value
g = -X'*(y./(1+exp(yXw))); % Gradient
end
