
function [U W] = altminGrpMTL(X, Y, W0, U0, rho, opts)

% X: 			centered data: cell array size K, each element Nt by J
% Y: 			cell array size K, each element vector size Nt
% rho: 		size: N_nleaf. Has regularization param of each inner node. 

[Tpa K] = size(U);
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
	U = updateU(W, U, rho, opts.eta);
	disp('Finished updating U....');

	% infer Tparents
	[vals idx] = max(U);
	U = zeros(size(U));
	U(sub2ind(size(U), idx, [1:K])) = 1;
	fprintf('Finished inferring parents ....\nNow calling GroupLasso to update Ws...\n');

Wold=W;
	% update W
  W = sqGroupLasso(W, Y, X, XX, XY, U, lambda, rho); 

%subplot(2,1,1);
%imagesc(W');
%subplot(2,1,2);
%imagesc(U);
%colormap(gray);
norm(W-Wold,'fro')
%pause;


	% print objective
  for task=1:K
    task_obj = sum(sum((Y{task} - X{task}*W(:,task)).^2))/2;
		%fprintf('[Task %d] sq.err: %f\n',task,task_obj);
    obj(iter) = obj(iter) + task_obj;
  end
	grpnorm = getGrpnorm(W, U, rho);
	fprintf('Iter: %d  lsqerr: %f  reg: %f   ',iter,obj(iter),grpnorm);
	obj(iter) = obj(iter) + lambda*grpnorm;
	fprintf('Obj: %f\n',obj(iter));
	disp('----------------------------');
end




