%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script: FCM test 
% Include : FCMClust(data, cluster_n, options)
% Author: 
% Date  :  
% Introduction : Divided into 3 categories 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 0: Clear Memory & Command Window
clc;
clear all;
close all;

%% Step 1: Make a random data
data = rand(100,2);
plot(data(:,1), data(:,2),'o');
hold on;

%% Step 2: FCM processing
[center,U,obj_fcn] = FCMClust(data,3);

%% Step 3: Find the results of Classification in U
maxU = max(U); % find the evey max element in every Column 
index1 = find(U(1,:) == maxU);
index2 = find(U(2,:) == maxU);
index3 = find(U(3,:) == maxU);

%% Step 4: Visualize results
line(data(index1,1),data(index1,2),'marker','*','color','g');
line(data(index2,1),data(index2,2),'marker','*','color','r');
line(data(index3,1),data(index3,2),'marker','*','color','b');
plot([center([1 2 3],1)],[center([1 2 3],2)],'*','color','k')
hold off;

%% Reference :
% [1] https://blog.csdn.net/lyxleft/article/details/88964494
% [2] https://www.cnblogs.com/wxl845235800/p/11053261.html

