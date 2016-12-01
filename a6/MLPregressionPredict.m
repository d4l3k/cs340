function [y] = MLPregressionPredict(Ww,X,nHidden)

[nInstances,nVars] = size(X);

% Form Weights
W1 = reshape(Ww(1:nVars*nHidden(1)),nVars,nHidden(1));
startIndex = nVars*nHidden(1);
for layer = 2:length(nHidden)
    Wm{layer-1} = reshape(Ww(startIndex+1:startIndex+nHidden(layer-1)*nHidden(layer)),nHidden(layer-1),nHidden(layer));
    startIndex = startIndex+nHidden(layer-1)*nHidden(layer);
end
w = Ww(startIndex+1:startIndex+nHidden(end));

h = @tanh; % Activation function

% Compute Output
y = zeros(nInstances,1);
for i = 1:nInstances
    innerProduct{1} = X(i,:)*W1;
    z{1} = h(innerProduct{1});
    for layer = 2:length(nHidden)
        innerProduct{layer} = z{layer-1}*Wm{layer-1};
        z{layer} = h(innerProduct{layer});
    end
    y(i) = z{end}*w;
end
