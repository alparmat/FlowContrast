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
%% avgFields: avg. velocity, pressure and vorticity in the mid longitudinal plane
%  Syntax:  type "avgFields" in MATLAB or Octave after the the first
%  output of the running simulation
%                                                                                
%  to enable this feature set delta=1 (line 5 in param.txt, after you have ran pipe.m)
%  to reset the averaging feature set delta=0 (line 5 in param.txt: 0)
%  
%  Output: Figure showing the average z-velocity, pressure and, x-vorticity
%          in the yz plane. The values are raw lattice (numerical) outputs
%          and have to be scaled to the actual incompressible flow problem
%
%
%  See also: plotyzplane.m - visualize the instantaneous flow field during
%            the run
%            avgRstress.m - visualize the averaged field fluctiations during
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
% load avreaged flow variables and reshape them
ux=load('_ux.txt'); uy=load('_uy.txt'); uz=load('_uz.txt'); p=load('_p.txt');

ux=reshape(ux,[length(ux)/lz lz]);
uy=reshape(uy,[length(uy)/lz lz]);
uz=reshape(uz,[length(uz)/lz lz]);
 p=reshape(p, [length(p )/lz lz]); 

ux=ux(:,(d+1):(lz-d)); 
uy=uy(:,(d+1):(lz-d));
uz=uz(:,(d+1):(lz-d));
 p=p(:,(d+1):(lz-d)); 
u=sqrt(ux.^2+uy.^2+uz.^2); 

% switch the wall region to (invisible) nan
p(wall)=nan; u(wall)=nan;

ax1=subplot(2,2,2);
pcolor(p); axis equal; shading interp; colormap jet; colorbar;  % pressure plot
title('avg. press. (numerical)')
% calculate and plot average vorticity
 ax3=subplot(2,2,3); 
 uzy=uz(2:end,:)-uz(1:end-1,:); u1=ux*0; u1(1:end-1,:)=uzy; uzy=u1;
 uyz=uy(:,2:end)-uy(:,1:end-1); u1=uy*0; u1(:,1:end-1)=uyz; uyz=u1;
 omega=uzy-uyz;omega(wall)=nan;
 pcolor((omega(:,1:end))); axis equal; shading interp; colormap(jet); colorbar; 
 title('avg. vort.-x (numerical)')
 
% plot average pressure at y=r 
 subplot(2,2,4);  plot(p(r,:))
 title('av. p. at y=r (num.)')
% plot average z-velocity 
 ax2=subplot(2,2,1); uz(wall)=nan; pcolor(-(uz(:,1:end)));
 axis equal; shading interp; colorbar;
 title('avg. vel.-z (numerical)')
linkaxes([ax1,ax2,ax3],'xy');

