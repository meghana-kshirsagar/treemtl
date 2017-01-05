
function genDataGrps(K, J, N, numClus)

X=cell(K,1);
Y=cell(K,1);
mu=zeros(J,1);
sigma=eye(J);
for t=1:K
	X{t} = mvnrnd(mu, sigma, N);
end

W=zeros(K,J);

fCount = J/numClus;
ntasks = K/numClus;
featIdx=0;
for clus=1:numClus
	stTask=(clus-1)*ntasks+1;
	endTask=clus*ntasks;
	w=0.5+rand(1,fCount);
	for tt=stTask:endTask
		W(tt,featIdx+1:featIdx+fCount) = (rand(1))*w;
	end
	featIdx=featIdx+fCount-0; % change here to produce overlapping clusters
end


figure;
imagesc(abs(W));
%colormap(gray);

W=W';

%Ntrain=70; Ntest=20; Nho=20;
CVO = cvpartition(N,'KFold',5);
perfold=N/5;

for t=1:K
	Y{t} = X{t}*W(:,t);
	data.X=X{t};
	data.Y=Y{t};
	save(sprintf('synthetic_data/30tasks_3clus_kang/task%d.mat',t),'data');
	for i = 1:CVO.NumTestSets
		split=i;
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    train=find(trIdx==1);
    test=find(teIdx==1);
		valid=train(end-perfold+1:end);
		train=train(1:end-perfold);
		save(sprintf('synthetic_data/30tasks_3clus_kang/task%d_split%d.mat',t,split),'train','valid','test');
	data.X=X{t}(train,:);
	data.Y=Y{t}(train,:);
	save(sprintf('synthetic_data/30tasks_3clus_kang/train_task%d_fold%d.mat',t,split),'data');
	data.X=X{t}(valid,:);
	data.Y=Y{t}(valid,:);
	save(sprintf('synthetic_data/30tasks_3clus_kang/ho_task%d_fold%d.mat',t,split),'data');
	data.X=X{t}(test,:);
	data.Y=Y{t}(test,:);
	save(sprintf('synthetic_data/30tasks_3clus_kang/test_task%d_fold%d.mat',t,split),'data');
	end
end

save(sprintf('synthetic_data/30tasks_3clus_kang/synth_W.mat'),'W');
