
function [Wnew] = treeGroupLasso(W, Y, X, T, Tw, XX, XY)

	% remove root node - last row
	T = T(1:end-1,:);
	Tw = Tw(1:end-1);
	% remove leaf nodes 
	idx=(sum(T,2)==1); 
	T(idx,:)=[];
	Tw(idx)=[];

  [V K] = size(T);
  sum_col_T=full(sum(T,2));
  SV=sum(sum_col_T);
  csum=cumsum(sum_col_T);
  g_idx=[[1;csum(1:end-1)+1], csum, sum_col_T]; %each row is the range of the group 
  P=zeros(SV,1);
  Q=zeros(SV,1);
  for v=1:V
     P(g_idx(v,1):g_idx(v,2))=find(T(v,:));
     Q(g_idx(v,1):g_idx(v,2))=Tw(v);
  end

	% SV: number of paths to leaves from all inner nodes. 
	% Each row of C is non-zero at the entry corresponding to the leaf it ends in
  C=sparse(1:SV, P, Q, SV, K); 

  TauNorm=repmat(Tw, 1, K).*T;
  TauNorm=max(sum(TauNorm.^2));
  L1=eigs(XX{1},1);

  option.maxiter=50;
  option.threshold=0;
  option.tol=1e-6;
	lambda=0.01; % regularization parameter
  mu=0.01;
  L=L1+lambda^2*TauNorm/mu;
  [Wnew, obj, time] = accgrad(W, Y, X, lambda, T,  XX, XY, C, g_idx, L, mu, option);


end
