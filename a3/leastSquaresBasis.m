function [model] = leastSquaresBasis(X,y,p)

X = polyBasis(X, p);

% Solve least squares problem
%w = (X'*X)\X'*y;
% we need to use inv instead due to "matrix singular to machine precision".
w = inv(X'*X)*X'*y;

model.w = w;
model.p = p;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
Xhat = polyBasis(Xhat, model.p);
w = model.w;
yhat = Xhat*w;
end

function [Xpoly] = polyBasis(X, p)
Xpoly = zeros(rows(X), p+1);
for i = 0:p
  Xpoly(:,i+1) = X.^i;
end
end
