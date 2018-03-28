%%
% File: l1emd_test.m
% Purpose:
% Calls the Julia function "wassersteindiscrete" indirectly via a system call,
% which runs the wrapper script "l1emd.jl". Interlanguage communication is
% achieved using temp files.
%
% Date: November 8, 2017
%%

mu = [0, 0];
Sig = eye(2);
m = 200;
n = 100;
XPts = mvnrnd(mu, Sig, m);
YPts = mvnrnd(mu, Sig, n);
SegY = [ones(n, 1), [1:n]'];
save('pts.mat', 'XPts', 'YPts', 'SegY');
system('julia l1emd.jl pts.mat emd.mat');
load('emd.mat');
plot(log(emd), 'linewidth', 1.5);
title('Sequentially computed EMD for IID sample');
ylabel('Log of EMD');
