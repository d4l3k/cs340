function [model] = knn(X,y,k)
  % [model] = knn(X,y,k)
  %
  % Implementation of k-nearest neighbour classifer

  model.X = X;
  model.y = y;
  model.k = k;
  model.c = max(y);
  model.predict = @predict;
end

function [yhat] = predict(model,Xtest)
  % compute euclidian distance
  % distances = sum((model.X - Xtest).**2, 2);
  [n,d] = size(model.X);
  [t,d] = size(Xtest);

  distances = model.X.^2*ones(d,t) + ones(n,d)*(Xtest').^2 - 2*model.X*Xtest';
  yhat = zeros(t, 1);
  for c = 1:t
    [dists,  idx] = sort(distances(:, c));
    yhat(c) = mode(model.y(idx(1:model.k, :)));
  end
end
