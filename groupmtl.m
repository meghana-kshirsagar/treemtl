
function [U W rho] = groupmtl(taskFiles, numClus)

K = length(taskFiles);
% load data
for t=1:K
	disp(taskFiles{t});
	load(taskFiles{t});
	X{t} = data.X;
	Y{t} = data.Y;
end
disp('Finished loading data..');

J = size(X{1},2);
W = zeros(J, K);
% center data and initialize Ws
for t=1:K
  [N] = size(X{t}, 1);
  Y{t} = (Y{t}-ones(N,1)*mean(Y{t},1));
  X{t} = (X{t}-ones(N,1)*mean(X{t},1));
	%W(:,t) = X{t} \ Y{t};
	W(:,t) = ridge(Y{t}, X{t}, 1);
end
disp('Finished centering data, initializing single task Ws..');

U = [];
Tpa = numClus;
if(Tpa > K)
	error('Too few tasks for such a big tree\n');
	return;
end
% initialize U using clusters of W
U = zeros(Tpa, K);  % each row represents one Uv
clusIdx=kmeans(W',Tpa);
for clus=1:Tpa
	U(clus,:)=(clusIdx==clus);
end

subplot(2,1,1);
imagesc(W');
subplot(2,1,2);
imagesc(U);
colormap(gray);
W0=W;

% call altmin
opts=[];
opts.eta=0.01;
opts.maxiter=20;
rho = 0.1*ones(Tpa,1);
[U W] = altmin(X, Y, W, U, rho, opts);

figure;
subplot(2,1,1);
imagesc(W');
subplot(2,1,2);
imagesc(U);
colormap(gray);
norm(W-W0,'fro')
