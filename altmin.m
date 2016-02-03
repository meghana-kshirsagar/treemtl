
function [U W] = altmin(X, Y, W0, U0, opts)

% X: 			centered data: cell array size K, each element Nt by J
% Y: 			cell array size K, each element vector size Nt

U = U0;
W = W0;
[Tpa K] = size(U);

disp('Initializing ...');
% precompute XX and XY
for task=1:K
  XX{task} = X{task}'*X{task};
  XY{task} = X{task}'*Y{task};
end

	function fval = getFusion()
		fval=0;	
    for t=1:K
      fval = fval + sum( sum( (repmat(U(:,t),(t-1),1) - U(:,[1:t-1])).^2 )) ;
    end		
	end

%figure;
obj=zeros(1,opts.maxiter);
for iter=1:opts.maxiter
	
	% update U
	U = updateU_fusion(W, U, opts);
	fprintf('Finished updating U.... \n Now updating W...\n');

	Wold=W;
	% update W
  W = sqGroupLasso(W, Y, X, XX, XY, U, opts); 

	subplot(3,2,2*iter+1);
	imagesc(abs(W));
	subplot(3,2,2*iter+2);
	imagesc(U);
	colormap(gray);
	pause;
	fprintf('Change in W: %f\n',norm(W-Wold,'fro'));

	% print objective
  for task=1:K
    task_obj(task) = sum(sum((Y{task} - X{task}*W(:,task)).^2))/2;
		%fprintf('[Task %d] sq.err: %f\n',task,task_obj);
  end
	grpnorm = getGrpnorm(W, U, opts.rho, opts.norm);
	fusion = getFusion();
	fprintf('[GLOBAL] Iter: %d  lsqerr: %f  reg: %f  fusion: %f\n', iter, sum(task_obj), opts.lambda*grpnorm, opts.mu*fusion);
	fprintf('L1 norm: %f  L12norm: %f  Fro-norm: %f\n',sum(sum(abs(W))),sum(sqrt(sum(W.^2,2))),norm(W,'fro'));
	obj(iter) = sum(task_obj) + opts.lambda*grpnorm + opts.mu*fusion;
	fprintf('Obj: %f\n',obj(iter));
	disp('----------------------------');

end



end % -- end function
