# treemtl
Matlab code from our ECML 2017 paper titled "Learning task structure via sparsity grouped multitask learning", Meghana Kshirsagar, Eunho Yang, Aurélie C. Lozano

Running the code:
======================
groupmtl(taskNames,taskFiles, testFiles, numClus, params)

taskNames: cell array with names of tasks, will be used while displaying results
taskFiles: cell array with paths to csv files containing training data
testFiles: cell array with paths to csv files containing test data
numClus:   number of clusters to assume. Run with a value that is about half the number of tasks.
params:    struct containing parameters used in optimization

taskNames, taskFiles, testFiles are all of size 'number of tasks'

Example run:
------------
taskNames={'task-1','task-2'};

for t=1:2

taskFiles{t}=sprintf('data/task%d.csv',t);

testFiles{t}=sprintf('data/test%d.csv',t);

end

Running the code on the Merck dataset:
========================================

for t=1:15

   taskFiles{t}=sprintf('merckdata/scaled/sampled/task%d_train.csv',t);
   
   testFiles{t}=sprintf('merckdata/scaled/sampled/task%d_test.csv',t);
   
end

params=[];

taskNames=[1:15];

groupmtl_merck(taskNames,taskFiles, testFiles, 7, params)


