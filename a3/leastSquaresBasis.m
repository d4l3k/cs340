function [model] = leastSquaresBasis(X,y,p)

X = polyBasis(X, p);

% Solve least squares problem
w = (X'*X)\X'*y;

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
Xpoly = [];
for i = 0:p
  Xpoly = [Xpoly X.^i];
end
end
