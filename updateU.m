function [U] = updateU(W, U, opts)

[J K] = size(W); % J: number of features, K: number of tasks
Tpa = size(U,1);

Wsq = W.^2;
eta = opts.eta_U;
Uold = U;

%%%%%% outer loop %%%%%%
for iter=1:opts.maxiter_U
    grad_u=zeros(size(U));
    % compute gradient w.r.t Upt.. note: 0 is a subgradient
    for p = 1:Tpa
    		denom = sqrt(sum(power(repmat(sqrt(U(p,:)),J,1).*W, 2), 2));  % denom: Jx1
    		for t=1:K
    			numer = Wsq(:,t);  % numer: Jx1 
    			numer(denom==0)=0;
    			denom(denom==0)=1;
    			grad_u(p,t) = sum(numer ./ denom);
    		end
				if opts.norm == 'l1'
	    		grad_u(p,:) = grad_u(p,:) * opts.rho(p);
				else
	    		grad_u(p,:) = grad_u(p,:) * 2 * opts.rho(p) * getGrpnorm(W,sqrt(U(p,:)),1,'l1');
				end
    end
    
    U_new = U - eta*grad_u;
    
    % project columns of U_new to a simplex i.e sum(U_new[:,t]) = 1  (i.e each task should only appear in one group)
    for t = 1:K
    	U_new(:,t) = projsplx(U_new(:,t));
    end
    
    U = U_new;

		% print R(U)
		R(iter) = getGrpnorm(W,sqrt(U),opts.rho(1:Tpa),opts.norm);
		%fprintf('R(U): %f\n',R);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%

%imagesc(U);
%colormap(gray);
%pause;

plot([1:opts.maxiter_U],R);
%pause;

% infer parents
[vals idx] = max(U);
U = zeros(size(U));
U(sub2ind(size(U), idx, [1:K])) = 1;
disp('Finished inferring new parents ....');

norm(U-Uold,'fro')

imagesc(U);
colormap(gray);
%pause;


