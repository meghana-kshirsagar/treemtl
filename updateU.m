function [U] = updateU(W, U, opts)

[J K] = size(W); % J: number of features, K: number of tasks
Tpa = size(U,1);

Wsq = W.^2;
eta = opts.eta_U;

%%%%%% outer loop %%%%%%
for iter=1:opts.maxiter_U
    grad_u=zeros(size(U));
    % compute gradient w.r.t Upt.. note: 0 is a subgradient
    for p = 1:Tpa
    		denom = sqrt(sum(power(repmat(U(p,:),J,1).*W, 2), 2));  % denom: Jx1
    		for t=1:K
    			numer =  U(p,t)*ones(J,1) .* Wsq(:,t);  % numer: Jx1 
    			numer(denom==0)=0;
    			denom(denom==0)=1;
    			grad_u(p,t) = sum(numer ./ denom);
    		end
    		grad_u(p,:) = grad_u(p,:) * 2 * opts.lambda * getGrpnorm(W,U(p,:),opts.rho(p));
    end
    
    U_new = U - eta*grad_u;
    
    % project columns of U_new to a simplex i.e sum(U_new[:,t]) = 1  (i.e each task should only appear in one group)
    for t = 1:K
    	U_new(:,t) = projsplx(U_new(:,t));
    end
    
    U = U_new;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%

% infer parents
[vals idx] = max(U);
U = zeros(size(U));
U(sub2ind(size(U), idx, [1:K])) = 1;
disp('Finished inferring new parents ....');


%imagesc(U);
%colormap(gray);
%pause;

