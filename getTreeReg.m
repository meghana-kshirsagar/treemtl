
function regVal = getTreeReg(Tnleaf, rho, U, W)
  regVal=0;
	K = size(U,2);
  [Nnleaf Tpa] = size(Tnleaf);
  for p = 1:Nnleaf
    d = (Tnleaf(p,:)==1); % descendants of p
    M = W*diag(sum([U(d,:); zeros(1,K)]));
    regVal = regVal + rho(p)*sum(sqrt(sum(M.*M,2)));
  end
end

