function [model] = decisionForest(X,y,depth,nBootstraps)

% Fit model to each boostrap sample of data
for m = 1:nBootstraps
    Xbootstrap = X;
    ybootstrap = y;
    n = rows(X);
    for i = 1:n
        j = ceil(rand * n);
        Xbootstrap(i,:) = X(j,:);
        ybootstrap(i,:) = y(j,:);
    end
    model.subModel{m} = randomTree(Xbootstrap,ybootstrap,depth);
end

model.predict = @predict;

end

function [y] = predict(model,X)

% Predict using each model
for m = 1:length(model.subModel)
    y(:,m) = model.subModel{m}.predict(model.subModel{m},X);
end

% Take the most common label
y = mode(y,2);
end
