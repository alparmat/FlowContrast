close all; clear all;
% d is the wall region, l is the total lentgth, lx is the pipe diameter
d=10; l=300+2*d; lx=100;
% contains the masked domain
IO=zeros(lx,lx,l); 
[lx,ly,lz]=size(IO); ox=(lx-1)/2; oy=(ly-1)/2; r=ox;

% calculate the pipe domain
for i=1:lx, for j=1:ly, for k=1:lz
    IO(i,j,k)=((+(i-ox-1)^2+(j-oy-1)^2)<r^2)*125;
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
