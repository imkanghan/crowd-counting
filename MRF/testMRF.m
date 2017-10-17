mex MRF.cpp

clear all
clc

% p is a local counts matrix that gets from the regression network 
p = imread('p.pgm');

%options = single([DISC_K DATA_K LAMBDA]);
options = single([3500 1000 0.85]);% UCF
%options = single([105 200 1.0])% Shanghaitech Part_A
%options = single([200 200 8]);% Shanghaitech Part_B

MRF(p, options)