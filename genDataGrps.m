
function genDataGrps(K, J, N)

X=cell(K,1);
Y=cell(K,1);
mu=zeros(J,1);
sigma=eye(J);
for t=1:K
	X{t} = mvnrnd(mu, sigma, N);
end

W=zeros(K,J);

numClus=3;
fCount = J/numClus;
ntasks = K/numClus;
featIdx=0;
for clus=1:numClus
	W((clus-1)*ntasks+1:clus*ntasks,featIdx+1:featIdx+fCount)=0.5;
	featIdx=featIdx+fCount; %-20; % change here to produce overlapping clusters
end

%W = W + 0.10*randn(size(W));

imagesc(W');
colormap(gray);

W=W';

for t=1:K
	Y{t} = X{t}*W(:,t);
	data.X=X{t};
	data.Y=Y{t};
	save(sprintf('synthetic_data/30tasks_3clusters/synth_task%d.mat',t),'data');
end

save(sprintf('synthetic_data/30tasks_3clusters/synth_W.mat'),'W');
