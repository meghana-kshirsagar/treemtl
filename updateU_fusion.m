function [U change] = updateU_fusion(W, U, opts)

[J K] = size(W); % J: number of features, K: number of tasks
Tpa = size(U,1);

Wsq = W.^2;
eta = opts.eta_U;
Uold = U;

threshold = 1e-7;

%figure;
%%%%%% outer loop %%%%%%
for iter=1:opts.maxiter_U
    grad_u=zeros(size(U));
    % compute gradient w.r.t Upt.. note: 0 is a subgradient
    for p = 1:Tpa
    		denom = sqrt(sum(power(repmat(sqrt(U(p,:)),J,1).*W, 2), 2));  % denom: Jx1
				denomcpy = denom;
    		for t=1:K
    			numer = Wsq(:,t);  % numer: Jx1 
    			numer(denom==0)=0;
    			denom(denom==0)=1;
    			grad_u(p,t) = sum(numer ./ denom);
					%fprintf('(%d,%d) grad: %f .. %f %f - zeros: %f %f\n',p,t,grad_u(p,t),sum(numer),sum(denom),sum(numer==0),sum(denom==0));
    		end
				if opts.norm == 'l1'
	    		grad_u(p,:) = grad_u(p,:) * opts.rho(p);
				else
	    		grad_u(p,:) = grad_u(p,:) * 2 * opts.rho(p) * sum(denomcpy);    %getGrpnorm(W,sqrt(U(p,:)),1,'l1');
				end

        % do fusion update
        for t=1:K
          grad_u(p,t) = opts.lambda * grad_u(p,t) + opts.mu * ( 2 * sum(U(p,t)-U(p,[1:t-1])) + 2 * sum(U(p,[t+1:K])-U(p,t)) );
        end
    end

    U_new = U - eta*grad_u;

    % project columns of U_new to a simplex i.e sum(U_new[:,t]) = 1  (i.e each task should only appear in one group)
    for t = 1:K
    	U_new(:,t) = projsplx(U_new(:,t));
    end
    
    U = U_new;

		% print R(U)
		R(iter) = opts.lambda * getGrpnorm(W,sqrt(U),opts.rho(1:Tpa),opts.norm) + opts.mu * getFusion(U,K);
		if(iter==1 || mod(iter,100)==0)
			fprintf('Iter %d R(U): %f\n',iter,R(iter));
			%grad_u
		end

		if (iter>10 && (((R(iter)-R(iter-1))/R(iter-1) > 1e-4) || abs(R(iter)-R(iter-1))/R(iter-1) < threshold))
			fprintf('Breaking\n');
			break
		end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%

%U

%plot([1:iter],R(1:iter));
%pause;

% infer parents
%if opts.norm == 'l1'
	[vals idx] = max(U);
	U = zeros(size(U));
	U(sub2ind(size(U), idx, [1:K])) = 1;
	disp('Finished inferring new parents ....');
%end

U'

change=norm(U-Uold,'fro');
fprintf('Iter: %d, Change in U: %f\n',iter,change);

%figure;
%imagesc(U);
%colormap(gray);
%pause;


