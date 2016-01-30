
function [U W] = altmin(X, Y, W0, U0, opts)

% X: 			centered data: cell array size K, each element Nt by J
% Y: 			cell array size K, each element vector size Nt
% rho: 		size: N_nleaf. Has regularization param of each inner node. 

U = U0;
W = W0;
[Tpa K] = size(U);

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
	%save('workspace_good.mat','U','W','opts');
	U = updateU(W, U, opts);
	fprintf('Finished updating U.... \n Now updating W...');

	Wold=W;
	% update W
  W = sqGroupLasso(W, Y, X, XX, XY, U, opts); 

	%subplot(2,1,1);
	%imagesc(W');
	subplot(3,3,iter);
	imagesc(U);
	colormap(gray);
	norm(W-Wold,'fro')
	pause;

	% print objective
  for task=1:K
    task_obj(task) = sum(sum((Y{task} - X{task}*W(:,task)).^2))/2;
		%fprintf('[Task %d] sq.err: %f\n',task,task_obj);
  end
	grpnorm = getGrpnorm(W, U, opts.rho, opts.norm);
	fprintf('Iter: %d  lsqerr: %f  reg: %f   ',iter,sum(task_obj),opts.lambda*grpnorm);
	obj(iter) = sum(task_obj) + opts.lambda*grpnorm; % + opts.lambda*sum(sum(abs(W)));
	fprintf('Obj: %f\n',obj(iter));
	disp('----------------------------');
end




