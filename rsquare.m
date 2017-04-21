
function r2 = rsquare(predy,y)

r2 = 1 - sum((y - predy).^2)/sum((y - mean(y)).^2);
