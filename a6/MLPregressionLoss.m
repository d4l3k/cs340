function [f,g] = MLPregressionLoss(Ww,X,y,nHidden)

[nInstances,nVars] = size(X);

% Form Weights
W1 = reshape(Ww(1:nVars*nHidden(1)),nVars,nHidden(1)); % Weight matrix W(1)
startIndex = nVars*nHidden(1);
for layer = 2:length(nHidden) % Weight matrix W(m) for m > 1
  Wm{layer-1} = reshape(Ww(startIndex+1:startIndex+nHidden(layer-1)*nHidden(layer)),nHidden(layer-1),nHidden(layer));
  startIndex = startIndex+nHidden(layer-1)*nHidden(layer);
end
w = Ww(startIndex+1:startIndex+nHidden(end)); % Final weight vector 'w'

h = @tanh; % Activation function
dh = @(z) sech(z).^2; % Derivative of activiation function

% Use sigmoid for activation function instead.
%h = @sigmoid;
%dh = @(z) sigmoid(z).*(1-sigmoid(z));

% Initialize gradient vector
f = 0;
if nargout > 1
    gradInput = zeros(size(W1));
    for layer = 2:length(nHidden)
       gradHidden{layer-1} = zeros(size(Wm{layer-1}));
    end
    gradOutput = zeros(size(w));
end

% Compute Output
for i = 1:nInstances
    innerProduct{1} = X(i,:)*W1;
    z{1} = h(innerProduct{1});
    for layer = 2:length(nHidden)
        innerProduct{layer} = z{layer-1}*Wm{layer-1};
        z{layer} = h(innerProduct{layer});
    end
    yhat = z{end}*w;

    r = yhat-y(i);
    f = f + r^2;

    if nargout > 1
        dr = 2*r;
        err = dr;

        % Output Weights
        gradOutput = gradOutput + err*z{end}';

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            backprop = err*(dh(innerProduct{end}).*w');
            gradHidden{end} = gradHidden{end} + z{end-1}'*backprop;

            % Other Hidden Layers
            for layer = length(nHidden)-2:-1:1
                backprop = (backprop*Wm{layer+1}').*dh(innerProduct{layer+1});
                gradHidden{layer} = gradHidden{layer} + z{layer}'*backprop;
            end

            % Input Weights
            backprop = (backprop*Wm{1}').*dh(innerProduct{1});
            gradInput = gradInput + X(i,:)'*backprop;
        else
            % Input Weights
            gradInput = gradInput + err*X(i,:)'*(dh(innerProduct{end}).*w');
        end

    end

end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(Ww));
    g(1:nVars*nHidden(1)) = gradInput(:);
    startIndex = nVars*nHidden(1);
    for layer = 2:length(nHidden)
        g(startIndex+1:startIndex+nHidden(layer-1)*nHidden(layer)) = gradHidden{layer-1};
        startIndex = startIndex+nHidden(layer-1)*nHidden(layer);
    end
    g(startIndex+1:startIndex+nHidden(end)) = gradOutput;
end
end

function [z] = sigmoid(y)
z = ones(size(y))./(1+exp(-y));
end

