load dog.mat

function [Iquant] = quantizeImage(I,b)
  [height, width, d] = size(I);
  flattened = reshape(I, height*width, d);
  model = clusterKmeans(flattened, 2^b, 0);
  out = model.W(model.predict(model,flattened),:);
  Iquant = reshape(out, height, width, d);
end

images = [];
for b = [1 2 4 6]
  fprintf('Processing %d\n', b)
  images = [images quantizeImage(I, b)];
end
set (figure, 'paperposition', [0 0 columns(I)*4/240 rows(I)/240+0.5])
image(flipud(images)/255);
%image(I/255);
print -dpng 4.1.2.png

