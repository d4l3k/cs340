load highway.mat
k = 5;

[n,d] = size(X);

model = dimRedPCA(X,k);

Z = model.compress(model,X);
Xhat = model.expand(model,Z);

for i = 1:n
    image = [reshape(X(i,:),[64 64]) reshape(Xhat(i,:),[64 64]) reshape(255*(abs(X(i,:)-Xhat(i,:))>10),[64 64])];
     imagesc(image);colormap gray
    %pause
end
