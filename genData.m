
function genData(K, J, N)

X=cell(K,1);
Y=cell(K,1);
mu=zeros(J,1);
sigma=eye(J);
for t=1:K
	X{t} = mvnrnd(mu, sigma, N);
end

W=zeros(K,J);
treeHt=5; 
featIdx=1;
for h=1:treeHt
	nChild=2^(h-1);
	nTasks=K/nChild;
	fCount=h;
	for c=1:nChild
		startIdx=(c-1)*nTasks+1;
		endIdx=startIdx+nTasks-1;
		W(startIdx:endIdx, featIdx:featIdx+fCount-1) = 0.5;
		featIdx=featIdx+fCount;
	end
end
W(:,featIdx:featIdx+K-1) = 0.5*eye(K);

imagesc(W);
colormap(gray);

W=W';

for t=1:K
	Y{t} = X{t}*W(:,t);
	data.X=X{t};
	data.Y=Y{t};
	save(sprintf('synthetic_data/synth_task%d.mat',t),'data');
end

save(sprintf('synthetic_data/synth_W.mat'),'W');
