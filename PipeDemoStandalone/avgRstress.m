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
%% avgRstress: avg. velocity, pressure and vorticity in the mid longitudinal plane
%  Syntax:  
%           type "avgRstress" in MATLAB or Octave after the the first
%           output of the running simulation
%                                                                                
%  to enable this feature set delta=2 (line 5 in param.txt, after you have ran pipe.m)
%  to reset the averaging feature set delta to 1 or 0 (line 5 in param.txt)
%
%  Output: Figure showing the average z-velocity, pressure and, x-vorticity
%          in the yz plane. The values are raw lattice (numerical) outputs
%          and have to be scaled to the actual incompressible flow problem
%
%
%  See also: plotyzplane.m - visualize the instantaneous flow field during
%            the run
%            avgFields.m - visualize the averaged flow field during
%            the run

%%

clear all; close all
% d is the wall region, lz is the total lentgth, lx is the pipe diameter
d=10; lz=3*100+2*d; r=50;
% load and reshape the mask
IO=load('IO.txt'); 
IO=reshape(IO,120,lz);  
IO=IO(:,d+1:end-d);
% identify wall region
wall=(IO>=125); 
% load avreaged flow fluctuation variables and reshape them

ux2=load('_ux2.txt');
uy2=load('_uy2.txt');
uz2=load('_uz2.txt');
p2=load('_p2.txt'); 
ux2=reshape(ux2,[length(ux2)/lz lz]); 
uy2=reshape(uy2,[length(uy2)/lz lz]); 
uz2=reshape(uz2,[length(uz2)/lz lz]);

uxy=load('_uxy2.txt'); 
uxz=load('_uxz2.txt'); 
uyz=load('_uyz2.txt');
uxy=reshape(uxy,[length(uxy)/lz lz]); 
uxz=reshape(uxz,[length(uxz)/lz lz]); 
uyz=reshape(uyz,[length(uyz)/lz lz]);
p2=reshape(p2,[length(p2)/lz lz]);

 p2= p2(:,(d+1):(lz-d)); 
ux2=ux2(:,(d+1):(lz-d)); 
uy2=uy2(:,(d+1):(lz-d)); 
uz2=uz2(:,(d+1):(lz-d)); 
uxy=uxy(:,(d+1):(lz-d)); 
uxz=uxz(:,(d+1):(lz-d)); 
uyz=uyz(:,(d+1):(lz-d)); 

% switch the wall region to (invisible) nan
ux2(wall)=nan;uy2(wall)=nan;uz2(wall)=nan;
p2(wall)=nan;

% calculate avg. |avg. uy-uy|
ax1=subplot(2,2,1);
pcolor(ux2); axis equal; shading interp; colormap jet; colorbar;  % pressure plot
title('avg. x-vel fluctuation')
% calculate avg. |avg. ux-ux|
 ax3=subplot(2,2,2); 
 pcolor(uy2); 
 axis equal; shading interp; colormap jet; colorbar;  % pressure plot
 title('avg. y-vel fluctuation')
% calculate avg. |avg. uz-uz|
ax4= subplot(2,2,3);  
 pcolor(uz2);  axis equal; shading interp; colormap jet; colorbar;  % pressure plot
 title('avg. z-vel fluctuation')
% calculate avg. |avg. p-p|
 ax2=subplot(2,2,4); 
 pcolor(p2); axis equal; shading interp; colormap jet; colorbar;  % pressure plot
 axis equal; shading interp; colorbar;
 title('avg. p fluctuation')
linkaxes([ax1,ax2,ax3,ax4],'xy');
