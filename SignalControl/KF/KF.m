%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script: Kalman filter tracking temperature
% Include : None
% Author: syf
% Date  : 2020-4-26 
% Introduction : 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 0: Clear Memory & Command Window
clc;
clear all;
close all;

%% Step 1: initiate the "Expected" "Measured" "Covariance" "noise"
N=300;                          % total sampling point

CON = 25;                       % create Expected matrix
expValue = CON*ones(1,N);       % Expected matrix

y = 2^0.5 * randn(1,N) + CON;   % create Measured matrix

x = zeros(1,N);                 % Final KF estimated value
x(1) = 1;

p = 10;                         % Covariance matrix

Q = cov(randn(1,N));            % covariance of process excitation noise
R = cov(randn(1,N));            % covariance of Measuring noise 

%% Step 2: Kalman filter Main loop
for k = 2 : N                   % start from 2
x(k) = x(k - 1);                % k times Predictive value
p = p + Q;                      % k times Covariance
kg = p / (p + R);               % kalman gain
x(k) = x(k) + kg*(y(k) - x(k)); % k times Final KF estimated value
p = (1 - kg) * p;               % update the Covariance
end

%% Step 3: Smooth part (Not research for now)
Filter_Wid = 10;
smooth_res = zeros(1,N);
for i = Filter_Wid + 1 : N
tempsum = 0;
for j = i - Filter_Wid : i - 1
tempsum = tempsum + y(j);
end
smooth_res(i) = tempsum / Filter_Wid;
end
t=1:N;
figure(1);
expValue = zeros(1,N);
for i = 1: N
expValue(i) = CON;
end

%% Step 3: draw the picture 
plot(t,expValue,'r',t,x,'g',t,y,'b',t,smooth_res,'k');
legend('real temperature','kalman result','measured value','smooth result');
axis([0 N 20 30])
xlabel('Sample Time');
ylabel('Room Temperature');
title('Smooth Filter VS Kalman Filter');

%% Reference :
% [1] explain KF
% https://blog.csdn.net/lybaihu/article/details/54943545
% [2] code come from
% https://blog.csdn.net/u013453604/article/details/50301477
% [3] Temperature example
% https://www.zhihu.com/question/22422121

