function [model] = weightedLeastSquares(X,y,z)

% Solve least squares problem
%w = (X'*z*X)\(X'*z*y)
w = 1/2*( (X'*z*X)\(y'*z*X) + X\y)

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
w = model.w;
yhat = Xhat*w;
end
