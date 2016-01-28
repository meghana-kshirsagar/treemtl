function [U_new] = updateU(W, U, Tnleaf, rho, eta)

%T: sparse matrix encoding tree; rows: number of non-leaf nodes, cols: number of tasks
%Tnleaf: rows: number of non-leaf nodes, cols: number of parents (|Tpa|)

[J K] = size(W); % J: number of features, K: number of tasks
Tpa = size(Tnleaf,2);

Wsq = W.*W;
%l12=sum(sqrt(sum(Wsq,2)))

%%%%%% outer loop %%%%%%
for iter=1:100

grad_u=zeros(size(U));
% compute gradient w.r.t Upt.. note: 0 is a subgradient
for p = 1:Tpa
	anc = find(Tnleaf(:,p)==1); % ancestors of p
	for aa = 1:numel(anc)
		desc = find(Tnleaf(anc(aa),:)==1); % descendants of a
		denom = sqrt(sum(power(repmat(sum([U(desc,:); zeros(1,K)]),J,1).*W, 2), 2));  % denom: Jx1 (note: outer sum is over K tasks)
		for t=1:K
			numer = sum(U(desc,t))*ones(J,1) .* Wsq(:,t);  % numer: Jx1 
			numer(denom==0)=0;
			denom(denom==0)=1;
			grad_u(p,t) = grad_u(p,t) + rho(anc(aa)) * sum(numer ./ denom);
		end
	end
end

U_new = U - eta*grad_u;
% project columns of U_new to a simplex i.e sum(U_new[:,t]) = 1  (i.e each task should only appear in one group)
for t = 1:K
	U_new(:,t) = projsplx(U_new(:,t));
end

U = U_new;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%

%imagesc(U_new);
%colormap(gray);
%pause;



