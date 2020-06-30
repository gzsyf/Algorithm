function [center, U, obj_fcn] = FCMClust(data, cluster_n, options)
% Using fuzzy C-means to cluster the data set into cluster_n.
% Brief:
% Parameters:
%	* data: n samples, each sample has m-dimensional eigenvalues 
%           (n*m matrix)
%	* N_cluster:  the number of aggregation centers,
%                 the number of categories (1*1 matrix)
%   * options:  4*1 matrix (optional)
%               options(1) -> The exponent of the membership matrix U
%               options(2) -> The maximum number of iterations
%               options(3) -> Minimum change in membership, iteration 
%                             termination condition
%               options(4) -> Whether to output information flags at
%                             each iteration
% Return:
%   * center: Clustering center
%   * U: Membership matrix
%   * obj_fcn:Objective function value
% Example:
%		>> [center,U,obj_fcn] = FCMClust(data,3);

%% step0: Processing input
% Description: 
% Attention:
if nargin ~= 2 && nargin ~= 3    
	error('Too many or too few input arguments!');
end

% default options Parameters
default_options = [2;	% The exponent of the membership matrix U
    100;                % The maximum number of iterations
    1e-5;               % termination condition
    1];                 % use output information flags


if nargin == 2          % Situation without nargin
	options = default_options;
else                    % Situation have nargin         
	if length(options) < 4  % parameter not enough  
		tmp = default_options;
		tmp(1:length(options)) = options;
		options = tmp;
    end
    % 
	nan_index = find(isnan(options)==1);
    %
	options(nan_index) = default_options(nan_index);
	if options(1) <= 1 % exponent of fuzzy matrix is less than one 
		error('The exponent should be greater than 1!');
	end
end
% put the values into Corresponding parameter 
expo = options(1);          % The exponent of the membership matrix U
max_iter = options(2);		% The maximum number of iterations
min_impro = options(3);		% Minimum change in membership, 
                            % iteration termination condition
display = options(4);		% Whether to output information flags 
                            % at each iteration

%% step1: 
% Description: 
% Attention:
data_n = size(data, 1); % data row - number of sample
in_n = size(data, 2);   % data columns - number of Eigenvalues

obj_fcn = zeros(max_iter, 1);	% initial obj_fcn
U = initfcm(cluster_n, data_n); % Initialize fuzzy assignment matrix

% Main loop  
for i = 1:max_iter
    % update cluster and membership 
	[U, center, obj_fcn(i)] = stepfcm(data, U, cluster_n, expo);
	if display
		fprintf('FCM:Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
	end
	% judgement
	if i > 1
        % if object function donot change
		if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro 
            break;
        end
	end
end

iter_n = i;	% 
obj_fcn(iter_n+1:max_iter) = [];



function U = initfcm(cluster_n, data_n)
% Brief: Initialize the membership function matrix of fcm
% Parameters: randomly initiate membership matrix 
%   * cluster_n:  Number of cluster centers
%   * data_n: number of Sample points
% Return:
%   * U: Initialized membership matrix           
U = rand(cluster_n, data_n);    % because every point have their membership 
                                % of every cluster so U=cluster_n*data_n
col_sum = sum(U); 
U = U./col_sum(ones(cluster_n, 1), :);


function [U_new, center, obj_fcn] = stepfcm(data, U, cluster_n, expo)
% Brief: Iterative step in fuzzy C-means clustering
% Parameters:
%   * data: nxm matrix, representing n samples, 
%           each sample has m eigenvalues      
%   * U: Membership matrix      
%   * cluster_n: Scalar, representing the number of clustering centers, 
%                the number of categories
%   * expo: The exponent of the membership matrix U           
% Return:
%   * U_new: new membership matrix    
%   * center: new cluster center    
%   * obj_fcn: Objective function value    
mf = U.^expo;       % ?
center = mf*data./((ones(size(data, 2), 1)*sum(mf'))'); % update cluster
                                                        % center 
dist = distfcm(center, data);       % calculate distance
obj_fcn = sum(sum((dist.^2).*mf));  % 
tmp = dist.^(-2/(expo-1));     
U_new = tmp./(ones(cluster_n, 1)*sum(tmp));  % 


function out = distfcm(center, data)
% Brief: Calculate the distance between the sample point and 
%        the cluster center
% Parameters:
%   center: cluster center    
%   data: sample point   
% Return:
%   out: distance       
out = zeros(size(center, 1), size(data, 1));
for k = 1:size(center, 1) % 
    % Euclid distance
    out(k, :) = sqrt(sum(((data-ones(size(data,1),1)*center(k,:)).^2)',1));
end

%% Reference :
% [1] https://blog.csdn.net/lyxleft/article/details/88964494
% [2] https://www.cnblogs.com/wxl845235800/p/11053261.html
