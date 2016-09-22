function [model] = decisionTreeHand(X,y,maxDepth)
    model.predict = @predict;
end

function [y] = predict(model,X)
  [t,d] = size(X);
  y = zeros(t,1);
  for i = 1:t;
    v = 0;
    if X(i,2) >= 37.695
      if X(i,1) >= -96.033
        v = 1;
      else
        v = 2;
      end
    else
      if X(i,1) >= -112.55
        v = 2;
      else
        v = 1;
      end
    end
    y(i, 1) = v;
  end
end

