load basisData.mat
[n,d] = size(X);

% Choose network structure
nHidden = [3];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end);
w = randn(nParams,1);

% Train with stochastic gradient
maxIter = 10000;
stepSize = 1e-4;
funObj = @(w,i)MLPregressionLoss(w,X(i,:),y(i),nHidden);
for t = 1:maxIter

    % The actual stochastic gradient algorithm:
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w = w - stepSize*g;
        
    % Every few iterations, plot the data/model:
    if mod(t-1,round(maxIter/100)) == 0
        fprintf('Training iteration = %d\n',t-1);
        figure(1);clf;hold on
        Xhat = [-10:.05:10]';
        yhat = MLPregressionPredict(w,Xhat,nHidden);
        plot(X,y,'.');
        h=plot(Xhat,yhat,'g-','LineWidth',3);
        drawnow;
    end
    
end