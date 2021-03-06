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
%% plotyzplane: flow velocity, pressure and vorticity in the mid longitudinal plane
%  Syntax:  type "plotyzplane" in MATLAB or Octave after the the first
%  output of the running simulation
%
%  Input:  	                                                                                  
%
%  Output: Figure showing the z-velocity, pressure and, x-vorticity in the 
%          yz plane. The values are raw lattice (numerical) outputs and
%          have to be scaled to the actual incompressible flow problem
%
%  Example: 	                                                                                  
%
%  See also: pipe.m - input for the simulation, FSLBM.cu - calculation source 	                                                                                  
%            avgFields.m -  visualize the averaged flow field during
%            the run

%%

% you can use this file to monitor the run in progress, all domain sizes
% must be consistent 
clear all; close all

% set the dimensions for the domain of interst
d=10; lz=100*3+2*d; r=50; 

% read the mask for the domain (y-z plane output from the run)
IO=load('IO.txt');
IO=reshape(IO,100+2*d,lz);  IO=IO(:,d+1:end-d);

% load velocity and pressure, reshape them
load('ux.txt'); load('uy.txt'); load('uz.txt'); load('p.txt');
ux=reshape(ux,[length(ux)/lz lz]);
uy=reshape(uy,[length(uy)/lz lz]);
uz=reshape(uz,[length(uz)/lz lz]);
 p=reshape(p, [length(p )/lz lz]);
lz=lz-d;
%reduce size
ux=ux(:,(d+1):lz); uy=uy(:,(d+1):lz); uz=uz(:,(d+1):lz); p=p(:,(d+1):lz); 
u=sqrt(ux.^2+uy.^2+uz.^2); zero=find(u==0.0); wall=(IO==125); 
ax1=subplot(2,2,2); 
% set the wall regions to invisible
p(wall)=nan; u(wall)=nan;ux(wall)=nan; uy(wall)=nan; uz(wall)=nan; 
% plot the numerical pressure
pcolor(p); axis equal; shading interp; colormap jet; colorbar;  
title('pressure (p) (numerical)')
ax3=subplot(2,2,3); 
% calculate in-plane vorticity
uzy=uz(2:end,:)-uz(1:end-1,:); u1=ux*0; u1(1:end-1,:)=uzy; uzy=u1;
uyz=uy(:,2:end)-uy(:,1:end-1); u1=ux*0; u1(:,1:end-1)=uyz; uyz=u1;
% set vorticity at the wall to invisible
omega=uzy-uyz;omega(wall)=nan;
%plot the vorticity
pcolor((omega(:,1:end))); axis equal; shading interp; colormap(jet); colorbar; %caxis([-0.02 0.02])
title('\omega_x vorticity (numerical)')
subplot(2,2,4);  plot(p(r,:))
title('p at y=r (numerical)')
% plot the z-direction velocity
ax2=subplot(2,2,1); uz(wall)=nan; pcolor(-(uz(:,1:end))); axis equal; shading interp; colorbar
title('velocity-z (numerical)')
linkaxes([ax1,ax2,ax3],'xy');
