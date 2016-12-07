load basisData.mat

[n,d] = size(X);

% Choose network structure
nHidden = [200 200];

% Train with stochastic gradient
maxIter = 30000;
stepSize = 1e-5;
batchSizeRandom = 3;
batchSizeWorst = 0;
batchRate = 0;%(n-batchSize)/maxIter;
momentum = 0.2;
tscale = 0.001;


% Normalize data
global centerX = mean(X);
global scaleX = 1/sqrt(var(X));
global centerY = mean(y);
global scaleY = 1/sqrt(var(y));

%var(X)
%var(y)

function [X, y] = transform(X, y)
  global centerX;
  global scaleX;
  global centerY;
  global scaleY;
  X = (X-centerX)*scaleX;
  y = (y-centerY)*scaleY;
end

[X, y] = transform(X, y);
[Xtest, ytest] = transform(Xtest, ytest);

minX = min(X)
maxX = max(X)


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

function n = randnormal(n, m, stdev)
  n = stdev.*randn(n,1) + m;
end

% Count number of parameters and initialize weights 'w', and learning rates
nParams = d*nHidden(1);
w = randnormal(nParams, 0, 1/sqrt(d));
learningRates = ones(nParams,1)*sqrt(d);
for h = 2:length(nHidden)
  connectionsIn = nHidden(h-1)
  added = connectionsIn*nHidden(h)
  w = [w; randnormal(added, 0, 1/sqrt(connectionsIn))];
  learningRates = [learningRates; ones(added, 1)*sqrt(connectionsIn)*sqrt(h)];
  nParams = nParams+added;
end
nParams = nParams+nHidden(end);
w = [w; randnormal(nHidden(end), 0, 1/sqrt(nHidden(length(nHidden)-1)))];
learningRates = [learningRates; ones(nHidden(end), 1)*sqrt(nHidden(length(nHidden)-1))];

w = zeroBasisRow(w, d, nHidden);
prevw = w;

bestw = [];
bestwscore = inf;

errors = ones(n,1)*inf;

funObj = @(w,i)MLPregressionLoss(w,applyBasis(X(i,:)),y(i),nHidden);
for t = 1:maxIter

    % The actual stochastic gradient algorithm:

    i = [];

    % Random mini-batches.
    i = [i ceil(rand(batchSizeRandom)*n)];

    % Worst mini-batches.
    if batchSizeWorst > 0
      [_, worst] = sort(errors(:),'descend');
      i = [i worst(1:batchSizeWorst, :)];
    end

    batchSizeRandom += batchRate;
    batchSizeWorst += batchRate;
    [f,g] = funObj(w,i);
    g = zeroBasisRow(g, d, nHidden);
    alphat = stepSize/sqrt(1+tscale*(t-1));
    betat = momentum;
    tmpw = w;
    w = w - alphat*g.*learningRates + betat*(w-prevw);
    prevw = tmpw;

    if batchSizeWorst > 0
      yhat = MLPregressionPredict(w,applyBasis(X(i,:)),nHidden);
      errors(i) = (yhat - y(i,:)).^2;
  end

    % Every few iterations, plot the data/model:
    if mod(t-1,round(maxIter/50)) == 0
        figure(1);clf;hold on
        Xhat = [minX:.05:maxX]';
        yhat = MLPregressionPredict(w,applyBasis(Xhat),nHidden);
        plot(X,y,'.');
        h=plot(Xhat,yhat,'g-','LineWidth',3);
        drawnow;

        % Compute training error
        yhat = MLPregressionPredict(w,applyBasis(X),nHidden);
        trainError = sum((yhat - y).^2)./rows(y);

        if trainError < bestwscore
          bestw = w;
          bestwscore = trainError;
        end

        fprintf('Training iteration = %d\n',t-1);
        fprintf('Training error = %.4f\n',trainError);
        fprintf('Alpha = %f, Beta = %f, W = %f\n', alphat, betat, sum(abs(w)));
    end

end

figure(1);clf;hold on
Xhat = [minX:.05:maxX]';
yhat = MLPregressionPredict(bestw,applyBasis(Xhat),nHidden);
plot(X,y,'.');
h=plot(Xhat,yhat,'g-','LineWidth',3);
drawnow;

% Compute test error
yhat = MLPregressionPredict(bestw,Xtest,nHidden);
testError = sum((yhat - ytest).^2)./rows(ytest);
fprintf('Test error = %.2f\n',testError);

print -dpng ./2.png
