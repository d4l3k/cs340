load animals.mat

k = 5;

[n,d] = size(X);
X = standardizeCols(X);

figure(1);
imagesc(X);
figure(2);
i = 1
j = 2

model = dimRedPCA(X,k);

Z = model.compress(model,X);

plot(Z(:,i),Z(:,j),'.');

for k = 1:n
		text(Z(k,i),Z(k,j),animals(k,:));
end

print -dpng 1.2.png

k
d
Xhat = model.expand(model,Z);
variance = norm(Xhat-X, "fro")^2/(norm(X, "fro")^2)
(1-variance)*100
