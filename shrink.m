function R=shrink(A, g_idx) % projects A onto an l2 ball of norm 1
    V=size(g_idx,1);
    R=zeros(size(A));
    for v=1:V
        idx=g_idx(v,1):g_idx(v,2);
        gnorm=sqrt(sum(A(idx,:).^2)); % l2 norm of group.. for each feature (gnorm: dx1)
        gnorm(gnorm<1)=1;  % do nothing for entries with norm<1.. if norm>1, then normalize to 1
        R(idx,:)=A(idx,:)./repmat(gnorm,  g_idx(v,3), 1);
    end
end
