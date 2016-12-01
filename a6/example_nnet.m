load basisData.mat
[n,d] = size(X);

d+=1

% Choose network structure
nHidden = [6];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end);
w = randn(nParams,1);
prevw = w;

function xb = applyBasis(X)
  xb = [ones(rows(X),1) X];
end

% Train with stochastic gradient
maxIter = 20000;
stepSize = 2e-3;
funObj = @(w,i)MLPregressionLoss(w,applyBasis(X(i,:)),y(i),nHidden);
for t = 1:maxIter

    % The actual stochastic gradient algorithm:
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    alphat = stepSize/sqrt(t);
    betat = 0.0;
    tmpw = w;
    w = w - alphat*g + betat*(w-prevw);
    prevw = tmpw;

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
        fprintf('Alpha = %f, Beta = %f, W = %f\n', alphat, betat, sum(abs(w)));
    end

end

% Compute test error
yhat = MLPregressionPredict(w,Xtest,nHidden);
testError = sum((yhat - ytest).^2)./rows(ytest);
fprintf('Test error = %.2f\n',testError);

print -dpng ./2.png