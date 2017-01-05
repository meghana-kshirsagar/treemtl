
function [Wnew] = sqGroupLasso(W, Y, X, XX, XY, U, opts)
  opts.threshold=0;
  opts.tol=1e-6;
	%opts.eta=5*1e-5;
	opts.eta=1e-5;
	g_idx = U;
  %[Wnew, obj, time, iter] = subgradDes(W, Y, X, XX, XY, g_idx, opts);
  [Wnew, obj, iter] = coordinateProx(W, Y, X, XX, XY, g_idx, opts);
end
