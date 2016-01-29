
function [U W] = altmin(X, Y, T, Tnleaf, W0, U0, rho, opts)

% Note:		K: number of tasks, N_nleaf: number of non-leaf nodes (K-1 for binary tree)
% X: 			centered data: cell array size K, each element Nt by J
% Y: 			cell array size K, each element vector size Nt
% T: 			sparse matrix encoding tree; rows: #total nodes, cols: K
% Tnleaf: nonleaf tree. rows: N_nleaf, cols: number of parents (|Tpa| or number of Us)
% rho: 		size: N_nleaf. Has regularization param of each inner node. 

% Note that Tnleaf will remain fixed throughout. T will change as the assignments of tasks to Tpar changes

[Nnodes K] = size(T);
[Nnleaf Tpa] = size(Tnleaf);
U = U0;
W = W0;

disp('Initializing ...');
% precompute XX and XY
for task=1:K
  XX{task} = X{task}'*X{task};
  XY{task} = X{task}'*Y{task};
end

figure;
obj=zeros(1,opts.maxiter);
for iter=1:opts.maxiter
	
	% update U
	U = updateU(W, U, Tnleaf, rho, opts.eta, opts.lambda);
	disp('Finished updating U....');

	% infer Tparents
	[vals idx] = max(U);
	U = zeros(size(U));
	U(sub2ind(size(U), idx, [1:K])) = 1;
	% update T
	T(K+1:K+Tpa, :) = U;
	for tidx=Tpa+1:Nnleaf
		T(K+tidx,:) = (sum(U( (Tnleaf(tidx,:)==1), : )) > 0);
	end
	fprintf('Finished inferring parents and updating tree T ....\nNow calling TreeGroupLasso to update Ws...\n');

Wold=W;
	% update W
  W = treeGroupLasso(W, Y, X, T, [opts.rhoLeaf*ones(K,1); rho], XX, XY, opts.lambda); % pad rhos with weights for leaves

subplot(2,1,1);
imagesc(W');
subplot(2,1,2);
imagesc(U);
colormap(gray);
norm(W-Wold,'fro')
pause;


	% print objective
  for task=1:K
    task_obj = sum(sum((Y{task} - X{task}*W(:,task)).^2))/2;
		%fprintf('[Task %d] sq.err: %f\n',task,task_obj);
    obj(iter) = obj(iter) + task_obj;
  end
	treeReg = getTreeReg(Tnleaf, rho, U, W);
	fprintf('Iter: %d  lsqerr: %f  treereg: %f   ',iter,obj(iter),treeReg);
	obj(iter) = obj(iter) + opts.lambda*treeReg + opts.lambda*sum(sum(abs(W)));
	fprintf('Obj: %f\n',obj(iter));
	disp('----------------------------');
end




