function [model] = softmaxClassifier(X,y)

% Compute sizes
[n,d] = size(X);
k = max(y);

W = zeros(d*k,1); % Each column is a classifier

[f,g] = softmaxLoss(W,X,y,d,k);
[f2,g2] = autoGrad(W,@softmaxLoss,X,y,d,k);

if max(abs(g-g2) > 1e-4)
    fprintf('User and numerical derivatives differ:\n');
    [g g2]
else
    fprintf('User and numerical derivatives agree.\n');
end


maxFunEvals = 1000; % Maximum number of evaluations of objective
verbose = 1; % Whether or not to display progress of algorithm


W = findMin(@softmaxLoss,W,maxFunEvals,verbose,X,y,d,k);

model.W = reshape(W, d, k);
model.predict = @predict;
end

function [yhat] = predict(model,X)
    W = model.W;
    [~,yhat] = max(X*W,[],2);
end

function [f,g] = softmaxLoss(w,X,y,d,k)
[n,d] = size(X);

w = reshape(w, d, k);

f = 0; % value
for i = 1:n
  smt = sum(exp(w' * X(i,:)'));
  f += -w(:,y(i))'*X(i,:)' + log(smt);
end
f;

g = zeros(d, k); % gradient
for j = 1:d
    for c = 1:k
        tempg = 0;
        for i = 1:n
            if y(i) == c
                tempg += -X(i, j)';
            end
            smt = sum(exp(w'*X(i,:)'));
            tempg += exp(w(:,c)'*X(i,:)')*X(i,j)/smt;
        end
        g(j,c) = tempg;
    end
end
g = reshape(g, d*k, 1);
end
