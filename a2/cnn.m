function [model] = cnn(X,y,k)
  % [model] = cnn(X,y,k)
  %
  % Implementation of condensed-nearest neighbour classifer

  % Always add the first point as a start.
  model.X = X(1, :);
  model.y = y(1, :);
  model.k = k;
  model.c = max(y);
  model.predict = @predict;

  [n, d] = size(X);
  for i = 2:n
    if model.predict(model, X(i, :)) != y(i)
      model.X = [model.X; X(i, :)];
      model.y = [model.y; y(i)];
    end
  end
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
    yhat(c) = mode(model.y(idx(1:min(n, model.k), :)));
  end
end
