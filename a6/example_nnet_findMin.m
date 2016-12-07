load basisData.mat

% Choose network structure
nHidden = [6];

% Train with stochastic gradient
maxIter = 20000;
stepSize = 2e-3;
batchSize = 3;
momentum = 0.2;

[n,d] = size(X);

% Add bias variable to first layer.
d+=1;

% Helper functions
function xb = applyBasis(X)
  xb = [ones(rows(X),1) X];
end

function [w] = zeroBasisRow(w, d, nHidden)
  startIndex = d*nHidden(1);
  for layer = 2:length(nHidden)
    for column = 1:nHidden(layer)
      w(startIndex+(nHidden(layer-1))*column) = 0;
    end
    startIndex = startIndex+nHidden(layer-1)*nHidden(layer);
  end
end


% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end);
w = randn(nParams,1);
w = zeroBasisRow(w, d, nHidden);
prevw = w;



%funObj = @(w,i)MLPregressionLoss(w,applyBasis(X(i,:)),y(i),nHidden);

function [f, g] = loss(w,i,X,y,nHidden)
  [f, g] = MLPregressionLoss(w,applyBasis(X(i,:)),y(i),nHidden);
  g = zeroBasisRow(g, columns(X), nHidden);
end

funObj = @loss

for t = 1:maxIter

    % The actual stochastic gradient algorithm:
    i = ceil(rand(batchSize)*n);
    w = findMin(funObj, w, 3, 0, i, X, y, nHidden);

    % Every few iterations, plot the data/model:
    if mod(t-1,round(maxIter/100)) == 0
        fprintf('Training iteration = %d\n',t-1);
        figure(1);clf;hold on
        Xhat = [-10:.05:10]';
        yhat = MLPregressionPredict(w,applyBasis(Xhat),nHidden);
        plot(X,y,'.');
        h=plot(Xhat,yhat,'g-','LineWidth',3);
        drawnow;

        % Compute training error
        yhat = MLPregressionPredict(w,applyBasis(X),nHidden);
        trainError = sum((yhat - y).^2)./rows(y);
        fprintf('Training error = %.2f\n',trainError);
        fprintf('Alpha = %f, Beta = %f, W = %f\n', 0, 0, sum(abs(w)));
    end

end

% Compute test error
yhat = MLPregressionPredict(w,Xtest,nHidden);
testError = sum((yhat - ytest).^2)./rows(ytest);
fprintf('Test error = %.2f\n',testError);

print -dpng ./2.png
