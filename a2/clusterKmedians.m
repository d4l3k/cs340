function [model] = clusterKmedians(X,k,doPlot)
% [model] = clusterKmedians(X,k,doPlot)
%
% K-medians clustering

[n,d] = size(X);
y = ones(n,1);

% Choose random points to initialize median
W = zeros(k,d);
for k = 1:k
    i = ceil(rand*n);
    W(k,:) = X(i,:);
end

X2 = X.^2*ones(d,k);
while 1
    y_old = y;

    % Draw visualization
    if doPlot && d == 2
        clustering2Dplot(X,y,W)
    end

    % Compute (squared) Euclidean distance between each data point and each
    % median
    distances = X2 + ones(n,d)*(W').^2 - 2*X*W';

    % Assign each data point to closest median
    [~,y] = min(distances,[],2);

    % Draw visualization
    if doPlot && d == 2
        clustering2Dplot(X,y,W)
    end

    % Compute median of each cluster
    for k = 1:k
        tX = X(y==k,:);
        %size(tX)
        if rows(tX) != 0
          W(k,:) = median(tX,1);
        end
    end

    changes = sum(y ~= y_old);

    % Stop if no point changed cluster
    if changes == 0
        break;
    end
end

model.W = W;
model.y = y;
model.predict = @predict;
model.error = @error;
end

function [y] = predict(model,X)
[t,d] = size(X);
W = model.W;
k = size(W,1);

% Compute Euclidean distance between each data point and each median
X2 = X.^2*ones(d,k);
distances = sqrt(X2 + ones(t,d)*(W').^2 - 2*X*W');

% Assign each data point to closest median
[~,y] = min(distances,[],2);
end

function [e] = error(model,X)
  c = model.predict(model,X);
  e = sum(sum(abs(X-model.W(c,:))));
end
