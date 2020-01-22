%{
BSD 2-Clause License
Copyright (c) 2019, alparmat
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%}
%% pipe: writes the domain for the pipe and the numerical variables
%  Syntax:  type "pipe" in MATLAB or Octave after the the first
%  output of the running simulation
%                                                   
%
%  Output: pipe.txt - text file that contains the domain
%          param.txt - parameter file can be changed during the run
%
%
%  See also: plotyzplane.m - visualize the flow field during the run 	                                                                                  
%

%%
close all; clear all;
% d is the wall region, l is the total lentgth, lx is the pipe diameter
d=10; l=300+2*d; lx=100;
% initialize masked domain
IO=zeros(lx,lx,l); 
[lx,ly,lz]=size(IO); ox=(lx)/2; oy=(ly)/2; r=ox;

% calculate the pipe domain
for i=1:lx, for j=1:ly, for k=1:lz
    IO(i,j,k)=((+(i-ox)^2+(j-oy)^2)<=r^2)*125;
end, end, end
IO=(~IO)*125;

% mark the cells with 1 used for the periodic boundary conditions
IO1=IO(:,:,1); IO1(IO1==0)=1; 
c=[1:d l-d+1:l];
for i=1:length(c)
  IO(:,:,c(i))=IO1;
end
% set wall regions
IO2=ones(lx+2*d,lx+2*d,l)*125; 
% calculate final domain
IO2(d+1:end-d,d+1:end-d,:)=IO; IO=IO2;
clear IO1; clear IO2;
[lx,ly,lz]=size(IO);
% write pipe domain text file
IO=uint8(IO); fid = fopen('pipe.txt', 'wt'); fprintf(fid, '%d\n', IO); fclose(fid)
% generate numerical parameters file that contains the blend factor,
% numerical p-difference, numerical viscosity, coarse corrective stencil
% length, statistical average switch, and blend factor for the correction
% order
fid = fopen('param.txt', 'wt'); 
fprintf(fid, '%f\n%f\n%f\n%d\n%d\n', 1.0,0.003,40.06e-4,2,1,0.0); 
fclose(fid)
