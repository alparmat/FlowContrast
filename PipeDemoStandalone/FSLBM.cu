#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <ctime>
#include <fstream>  
#include <iostream>
#define pathlen 1000
# define bdx 8 // dimension of the x-block (lattice Boltzmann kernel)
# define bdy 8
# define bdz 8 
surface < void,  3 >                                    surf;
texture < float4, cudaTextureType3D, cudaReadModeElementType >  vtexr;
//nvcc FSLBM.cu -Xptxas -dlcm=cg -maxrregcount 64 -arch compute_61 -o pipe.exe
// ****************************** compute kernel ***************************
__global__ void 
FSLBMcalc(char* __restrict__ IO, int lx, int ly, int lz, float dtddy, float pin,
        float nu,int shift,int k,float tblend, int d1,float m,float blend){
//set up coordinates for usage with texture            
int lxy = lx * ly; 
// x and y are for a full plane from texture
int x = ( blockIdx.x * blockDim.x + threadIdx.x );   
int y = ( blockIdx.y * blockDim.y + threadIdx.y);   
// z is a range before k in the texture update direction
int z = k - int( threadIdx.z );
// exit the kernel if the coordinates are specified outside the domain
if (x >= lx || y >= ly || z >= lz || z<0 || y<0 || x<0 ) return;
// get linear coordinate from texture coordinate (for usage with the mask IO)
int idx = int( x + lx * y + z * lxy ); 
// thread adressing 
int tx=threadIdx.x; int ty=threadIdx.y; int tz= blockDim.z - threadIdx.z - 1;
// lattice Boltzmann weights and utility cell directions needed for all updates
const float W[9]={ 0.296296296f, 0.07407407407f, 0.07407407407f, 0.07407407407f,
        0.07407407407f, 0.01851851851f, 0.01851851851f, 0.01851851851f, 0.01851851851f };
const char cy[9]={0, 1,-1, 0, 0, 1,-1, 1,-1};
const char cz[9]={0, 0, 0, 1,-1, 1,-1,-1, 1};

//val will contain the final result for velocity and pressure
float4 val;
//utility variables during predictive step 
float rt1v, sw, zet_u_RTinv, u_u_RTinv;
//utility variables for corrective step
float uxt1v = 0.0f; float uyt1v = 0.0f; float uzt1v = 0.0f; float rt1 = 0.0f;             
//padding coordinates to be used in GPU shared memory
int tzp = tz+2; int typ = ty+2; int txp = tx+2; 
// padding coordinates to be used in the GPU global memory
int xp=x+1; int xm=x-1;  int yp=y+1; int ym=y-1;  int zp=z+1; int zm=z-1;
//coordiate for half - blocks
int bx=blockDim.x>>1; int by=blockDim.y>>1; int bz=blockDim.z>>1;
// shared memory use to facilitate efficient global memory acess 
// hydrodynamic variables (loaded from texture) are stored in the 3D shared memory padded-blocks
__shared__ float shp[bdz+2][bdy+2][bdx+2]; 
__shared__ float shu[bdz+2][bdy+2][bdx+2];
__shared__ float shv[bdz+2][bdy+2][bdx+2];
__shared__ float shw[bdz+2][bdy+2][bdx+2];

//fill up shared memory by use of padded coordinates in 8 directions
if(tx>=bx-1 && ty>=by-1 && tz>=bz-1){
val=tex3D(vtexr, xp, yp, zp);   //sub-block 1 (includes padding)
shp[tzp][typ][txp]=val.x; 
shu[tzp][typ][txp]=val.y; shv[tzp][typ][txp]=val.z; shw[tzp][typ][txp]=val.w; }

if(tx<=bx && ty>=by-1 && tz>=bz-1){
val=tex3D(vtexr, xm, yp, zp);   //sub-block 2 (includes padding)
shp[tzp][typ][tx]=val.x; 
shu[tzp][typ][tx]=val.y; shv[tzp][typ][tx]=val.z; shw[tzp][typ][tx]=val.w; }

if(tx<=bx && ty<=by && tz>=bz-1){
val=tex3D(vtexr, xm, ym, zp);   //sub-block 3 (includes padding) 
shp[tzp][ty][tx]=val.x; 
shu[tzp][ty][tx]=val.y; shv[tzp][ty][tx]=val.z; shw[tzp][ty][tx]=val.w; }

if(tx>=bx-1 && ty<=by && tz>=bz-1){
val=tex3D(vtexr, xp, ym, zp);   //sub-block 4 (includes padding) 
shp[tzp][ty][txp]=val.x; 
shu[tzp][ty][txp]=val.y; shv[tzp][ty][txp]=val.z; shw[tzp][ty][txp]=val.w; }

if(tx>=bx-1 && ty<=by && tz<=bz){
val=tex3D(vtexr, xp, ym, zm);   //sub-block 5 (includes padding) 
shp[tz][ty][txp]=val.x; 
shu[tz][ty][txp]=val.y; shv[tz][ty][txp]=val.z; shw[tz][ty][txp]=val.w; }

if(tx>=bx-1 && ty>=by-1 && tz<=bz){
val=tex3D(vtexr, xp, yp, zm);   //sub-block 6 (includes padding) 
shp[tz][typ][txp]=val.x;
shu[tz][typ][txp]=val.y;shv[tz][typ][txp]=val.z;shw[tz][typ][txp]=val.w; }

if(tx<=bx && ty>=by-1 && tz<=bz){
val=tex3D(vtexr, xm, yp, zm);   //sub-block 7 (includes padding) 
shp[tz][typ][tx]=val.x; 
shu[tz][typ][tx]=val.y; shv[tz][typ][tx]=val.z; shw[tz][typ][tx]=val.w; }

if(tx<=bx && ty<=by && tz<=bz){
val=tex3D(vtexr, xm, ym, zm);   //sub-block 8 (includes padding) 
shp[tz][ty][tx]=val.x; 
shu[tz][ty][tx]=val.y; shv[tz][ty][tx]=val.z; shw[tz][ty][tx]=val.w; }

//shared memory block, finished
txp=tx+1; typ=ty+1; tzp=tz+1; 
// finish read-shared memory  thread interactions
__syncthreads();
// update only the fluid region (no wall, no padded inlet/outlet update in this kernel)
if (IO[idx]==0){
//get pressure and velocity for current thread
float rhd1=shp[tzp][typ][txp]; 
float u=shu[tzp][typ][txp]; float v=shv[tzp][typ][txp]; float w=shw[tzp][typ][txp];
// perform predictive (lattice Boltzmann) step: d3q27 stencils are devided into 3 planes
for (int i=8;i>=0;i--){
    // check wether the next point is in the wall or not, sw contains info
   int id2=lx*int(cy[i])+int(cz[i])*lxy;         
   sw= float(IO[idx+id2]<125); 
   // read pressure & u,v,w neighbors from shared memory 
   val.x=shp[tzp+cz[i]][typ+cy[i]][txp]*sw+rhd1*(1.0f-sw);    
   val.y=shu[tzp+cz[i]][typ+cy[i]][txp];
   val.z=shv[tzp+cz[i]][typ+cy[i]][txp];
   val.w=shw[tzp+cz[i]][typ+cy[i]][txp];
   // start building parts of the equilibrium distribution function
   zet_u_RTinv=(float(cy[i])*val.z+float(cz[i])*val.w)*3.0f; 
   u_u_RTinv=(val.y*val.y+val.z*val.z+val.w*val.w)*3.0f;
   // calculate equilibrium df. while x=0 (mid plane)
   rt1v=W[i]*(val.x+zet_u_RTinv+0.5f*(zet_u_RTinv*zet_u_RTinv-u_u_RTinv)); 
   // compute predicted pressure and velocities
   rt1+=rt1v; uyt1v+=rt1v*float(cy[i]); uzt1v+=rt1v*float(cz[i]);          
   // same as above but for the x->x+1 direction
   sw= float(IO[idx+id2+1]<125);
   val.x=shp[tzp+cz[i]][typ+cy[i]][txp+1]*sw+rhd1*(1.0f-sw); 
   val.y=shu[tzp+cz[i]][typ+cy[i]][txp+1]; 
   val.z=shv[tzp+cz[i]][typ+cy[i]][txp+1]; 
   val.w=shw[tzp+cz[i]][typ+cy[i]][txp+1];
   zet_u_RTinv=(val.y+float(cy[i])*val.z+float(cz[i])*val.w)*3.0f; 
   u_u_RTinv=(val.y*val.y+val.z*val.z+val.w*val.w)*3.0f;
   rt1v=W[i]*(val.x+zet_u_RTinv+0.5f*(zet_u_RTinv*zet_u_RTinv-u_u_RTinv))*0.25f;  
// calculate equilibrium df. while x=+1
   rt1+=rt1v; uyt1v+=rt1v*float(cy[i]); uxt1v+=rt1v; uzt1v+=rt1v*float(cz[i]);
  
   // same as above but for the x->x-1 direction
   sw= float(IO[idx+id2-1]<125);
   val.x=shp[tzp+cz[i]][typ+cy[i]][txp-1]*sw+rhd1*(1.0f-sw); 
   val.y=shu[tzp+cz[i]][typ+cy[i]][txp-1]; 
   val.z=shv[tzp+cz[i]][typ+cy[i]][txp-1]; 
   val.w=shw[tzp+cz[i]][typ+cy[i]][txp-1];   
   zet_u_RTinv=(-val.y+float(cy[i])*val.z+float(cz[i])*val.w)*3.0f; 
   u_u_RTinv=(val.y*val.y+val.z*val.z+val.w*val.w)*3.0f;
   rt1v=W[i]*(val.x+zet_u_RTinv+0.5f*(zet_u_RTinv*zet_u_RTinv-u_u_RTinv))*0.25f; 
// calculate equilibrium df. while x=-1
   rt1+=rt1v; uyt1v+=rt1v*float(cy[i]); uxt1v-=rt1v; uzt1v+=rt1v*float(cz[i]);}

// select corrective step order 
// (currently a blend is used between order 4 and 6 finite difference correction)
// order 14 
/* float c0=-9.16453231293f; float c1=1.77777777778f; float c2=-0.31111111111f; 
   float c3=0.07542087542f; float c4=-0.01767676767f; float c5=3.48096348e-3f; 
   float c6=-5.1800052e-4f; float c7=5.0742908e-5f; float c8=-2.4281274e-6f;
*/

// order 12        
/* float c0=-8.94833333333f; float c1=1.71428571429f; float c2=-0.26785714285f; 
   float c3=0.05291005291f; float c4=-0.00892857142f; float c5=0.00103896103f;
   float c6=-0.00006012506;  float c7=0.0f; float c8=0.0f;
*/
        
// order 10        
/* float c0=-8.78166666666f; float c1=1.66666666f; float c2=-0.23809523809f; 
   float c3=0.03968253968f; float c4=-0.00496031746f; float c5=0.00031746031f; 
   float c6=0.0f; float c7=0.0f; float c8=0.0f;
*/

// order 8        
/* float c0=-8.541666666666666f; float c1=1.6f; float c2=-0.2f; 
   float c3=0.025396825396825f; float c4=-0.001785714285714; float c5=0.0f;  
   float c6=0.0f; float c7=0.0f; float c8=0.0f;
*/
        
// order 6        
/*float c0=-8.16666666f; float c1=1.5f; float c2=-0.15f; float c3=0.01111111f;
  float c4=0.0f; float c5=0.0f;   float c6=0.0f; float c7=0.0f; float c8=0.0f;
*/
        
// order 4        
/*float c0=-7.5f; float c1=1.3333333f; float c2=-0.08333333f; float c3=0.0f;
  float c4=0.0f; float c5=0.0f;   float c6=0.0f; float c7=0.0f; float c8=0.0f;
*/
        
// order 2       
/*float c0=-6.0f; float c1=1.0f; float c2=0.f; float c3=0.0f; float c4=0.0f;
  float c5=0.0f;  float c6=0.0f; float c7=0.0f; float c8=0.0f;
*/
        
// order 8 <-> 6
/*float blend =0.0f; float c0=-8.166666666666666f-0.375f*blend; float c1=1.5f+0.1f*blend; 
  float c2=-0.15f-0.05f*blend; float c3=0.01111111f+0.014285714285714f*blend; 
  float c4=-0.001785714285714f*blend; float c5=0.0f;   float c6=0.0f; 
  float c7=0.0f; float c8=0.0f;
*/

// order 6 <-> 4      
/*        
float c0=-7.5f-0.66666666666f*blend; float c1=1.3333333f+0.16666666666f*blend; 
float c2=-0.08333333f-0.0666666666666f*blend; float c3=0.01111111f*blend; 
//float c4=0.0f; float c5=0.0f;   float c6=0.0f; float c7=0.0f; float c8=0.0f;
*/              
// order 4 <-> 2        

float c0=-6.0f-1.5f*blend;  float c1=1.0f+0.3333333333f*blend;  float c2=-0.0833333333f*blend;
//float c3=0.0f; float c4=0.0f; float c5=0.0f;  float c6=0.0f; float c7=0.0f;

        
//start with defining corrective stencil midpoints
float uxt2=u*c0; float uyt2=v*c0; float uzt2=w*c0;

float4 uvw[6];
/* build corrective step stencils up to order 6, increasing to higher 
*  order needs code extension below (slower)
*/
// order 2        
uvw[0]=tex3D(vtexr,x+d1,y,z); uvw[1]=tex3D(vtexr,x-d1,y,z); 
uvw[2]=tex3D(vtexr,x,y+d1,z); uvw[3]=tex3D(vtexr,x,y-d1,z); 
uvw[4]=tex3D(vtexr,x,y,z+d1); uvw[5]=tex3D(vtexr,x,y,z-d1); 

uxt2+=c1*(uvw[0].y+uvw[1].y+uvw[2].y+uvw[3].y+uvw[4].y+uvw[5].y);
uyt2+=c1*(uvw[0].z+uvw[1].z+uvw[2].z+uvw[3].z+uvw[4].z+uvw[5].z);
uzt2+=c1*(uvw[0].w+uvw[1].w+uvw[2].w+uvw[3].w+uvw[4].w+uvw[5].w);
//order 4
uvw[0]=tex3D(vtexr,x+2*d1,y,z); uvw[1]=tex3D(vtexr,x-2*d1,y,z);
uvw[2]=tex3D(vtexr,x,y+2*d1,z); uvw[3]=tex3D(vtexr,x,y-2*d1,z);
uvw[4]=tex3D(vtexr,x,y,z+2*d1); uvw[5]=tex3D(vtexr,x,y,z-2*d1);

uxt2+=c2*(uvw[0].y+uvw[1].y+uvw[2].y+uvw[3].y+uvw[4].y+uvw[5].y);
uyt2+=c2*(uvw[0].z+uvw[1].z+uvw[2].z+uvw[3].z+uvw[4].z+uvw[5].z);
uzt2+=c2*(uvw[0].w+uvw[1].w+uvw[2].w+uvw[3].w+uvw[4].w+uvw[5].w);
//order 6 (3*d1 = coarse corrective stencil length)
/*        
uvw[0]=tex3D(vtexr,x+3*d1,y,z); uvw[1]=tex3D(vtexr,x-3*d1,y,z);
uvw[2]=tex3D(vtexr,x,y+3*d1,z); uvw[3]=tex3D(vtexr,x,y-3*d1,z);
uvw[4]=tex3D(vtexr,x,y,z+3*d1); uvw[5]=tex3D(vtexr,x,y,z-3*d1);

uxt2+=c3*(uvw[0].y+uvw[1].y+uvw[2].y+uvw[3].y+uvw[4].y+uvw[5].y);
uyt2+=c3*(uvw[0].z+uvw[1].z+uvw[2].z+uvw[3].z+uvw[4].z+uvw[5].z);
uzt2+=c3*(uvw[0].w+uvw[1].w+uvw[2].w+uvw[3].w+uvw[4].w+uvw[5].w);
*/
// perform corrective step
     float c=tblend; float cc=1.0-tblend;
       val.x=rt1;
       val.y=(uxt1v+dtddy*uxt2*m)*c+u*cc;      
       val.z=(uyt1v+dtddy*uyt2*m)*c+v*cc; 
       val.w=(uzt1v+dtddy*uzt2*m)*c+w*cc;
//  write new result as a surface
       surf3Dwrite(val,surf,x * sizeof(float4),y,z+float(shift)); 
}}

// ***************************** get planes *******************************
// switches the planes used for periodic boundary conditions, subtracts the outlet pressure 
__global__ void bcond(char *IO, long lx, long ly, long lz, long d1, float pin, float p2){
// coordinate in the texture     
int x = blockIdx.x * blockDim.x + threadIdx.x; 
int y = blockIdx.y * blockDim.y + threadIdx.y; 
int z = blockIdx.z * blockDim.z + threadIdx.z;
// coordinate in linear memory
int idx=int(x) + lx * int(y) + int(z) * lx * ly;
// return if coordinates are outside of the domain
if (x >= lx || y >= ly || z >= lz)  return;
// write 0 to all variables and return if wall 
if (IO[idx]==125) {surf3Dwrite(float4{0.0f,0.0f,0.0f,0.0f},surf,x * sizeof(float4),y,z);return;}
// build periodic regions
// before inlet copy the same variables as before outlet and add a p-difference        
 int sw=1;
 if ((z>=0) && (z<d1)){
 float4 val=tex3D(vtexr,x,y,lz-2*d1+z); val.x-=p2;
  val.x+=pin; 
 surf3Dwrite(val,surf,x * sizeof(float4),y,z); sw=0;} 
// after outlet copy the same variables as after inlet and subtract p-difference
 if ((z<lz) && (z>=lz-d1)){
 float4 val=tex3D(vtexr,x,y,2*d1-lz+z); val.x-=pin; val.x-=p2;
 surf3Dwrite(val,surf,x * sizeof(float4),y,z); sw=0; }
// subtract outlet pressure from additional points
if (sw==1) {float4 val=tex3D(vtexr,x,y,z); val.x-=p2; surf3Dwrite(val,surf,x * sizeof(float4),y,z);}}
       

//shift the k-th plane from z direction to z+d1
__global__ void shiftback(int lx, int ly, int lz, int d1, int k){
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y; 
int z = k;
if (x >= lx || y >= ly || z >= lz )  return;
float4 val=tex3D(vtexr,x,y,z+d1); surf3Dwrite(val,surf,x * sizeof(float4),y,z); }

//get a velocity component or pressure from the YZ plane with "0<=planeidx<lx"
__global__ void getYZplane(float *out, long lx, long ly, long lz, long planeidx, int type){
int x = blockIdx.x*blockDim.x + threadIdx.x;   
int y = blockIdx.y*blockDim.y + threadIdx.y;   
int z = blockIdx.z*blockDim.z + threadIdx.z;
if (x >= lx || y >= ly || z >= lz )  return;

if (x==planeidx){ 
float4 val=surf3Dread<float4>(surf,x * sizeof(float4),y,z);
out[y+ly*z]=val.x * float(type==0) + val.y * float(type==1) +
            val.z * float(type==2) + val.w * float(type==3);}}

//get a velocity component or pressure from the XY plane with "0<=planeidx<lz"
__global__ void getXYplane(float *out, long lx, long ly, long lz, long planeidx, int type){
int x = blockIdx.x*blockDim.x + threadIdx.x;   
int y = blockIdx.y*blockDim.y + threadIdx.y;   
int z = blockIdx.z*blockDim.z + threadIdx.z;
if (x >= lx || y >= ly || z >= lz )  return;

if (z==planeidx){
    float4 val=surf3Dread<float4>(surf,x * sizeof(float4),y,z);
out[x+lx*y]=val.x * float(type==0) + val.y * float(type==1) +
            val.z * float(type==2) + val.w * float(type==3);}}

//get a velocity component or pressure from the ZX plane with "0<=planeidx<ly"
__global__ void getZXplane(float *out, long lx, long ly, long lz, long planeidx, int type){
int x = blockIdx.x*blockDim.x + threadIdx.x;   
int y = blockIdx.y*blockDim.y + threadIdx.y;   
int z = blockIdx.z*blockDim.z + threadIdx.z;
if (x >= lx || y >= ly || z >= lz )  return;

if (z==planeidx) {
    float4 val=surf3Dread<float4>(surf,x * sizeof(float4),y,z);
out[x+lx*z]=val.x * float(type==0) + val.y * float(type==1) +
            val.z * float(type==2) + val.w*float(type==3);}}

//set a velocity component or pressure for the YZ plane with "0<=planeidx<lx"
__global__ void setXYplane(float *in, long lx, long ly, long lz, long planeidx, int type){
int x = blockIdx.x*blockDim.x + threadIdx.x;   
int y = blockIdx.y*blockDim.y + threadIdx.y;   
int z = blockIdx.z*blockDim.z + threadIdx.z;
if (x >= lx || y >= ly || z >= lz )  return;

if (z==planeidx) { 
    float4 val=surf3Dread<float4>(surf,x * sizeof(float4),y,z); float val1=in[x+lx*y];
    val.x=val.x*float(1-int(type==0))+float(type==0)*val1; 
    val.y=val.y*float(1-int(type==1))+float(type==1)*val1;
    val.z=val.z*float(1-int(type==2))+float(type==2)*val1; 
    val.w=val.w*float(1-int(type==3))+float(type==3)*val1;
    surf3Dwrite(val,surf,x * sizeof(float4),y,z);}}

// get YZ plane from mask with x=planeidx
__global__ void getYZplaneIO(char *in, char *out, long lx, long ly, long lz, long planeidx){
long idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < lx * ly * lz) {
long k = idx/lx/ly; long j = (idx-k*lx*ly)/lx; long i = idx-j*lx-k*lx*ly;   // get ijk coordinates
if (i==planeidx) {out[j+k*ly]=(in[idx]);}}}

//  **************************** init functions ***************************
// initialize all variables to 0        
__global__ void initphys(float pin, int lx, int ly, int lz, int d2){
int x = blockIdx.x*blockDim.x + threadIdx.x;   
int y = blockIdx.y*blockDim.y + threadIdx.y;   
int z = blockIdx.z*blockDim.z + threadIdx.z;
if (x >= lx || y >= ly || z >= lz )  return;

float4 val;
val.x = pin * float( lz-d2-z )/float( lz-2*d2 );       //set p
val.y = 0.0f; val.z = 0.0f; val.w = 0;
surf3Dwrite(val,surf,x * sizeof(float4), y, z);}

// alternatively read the variables from a previously saved run ( can be swithced instead of using the initphys kernel)
void initphys1(float *xytmp, float *xytmph, int lx, int ly, int lz, char *path){
  char path1[pathlen]; strcpy(path1,path);   
  dim3 blocks3D(bdx,bdy,bdz); 
  dim3 grid3D(lx/bdx+1,ly/bdy+1,lz/bdz+1);

//read pressure
FILE* file  = fopen (strcat(path1,"p3Dii"), "rb");  
printf("file : %s\n", path1); strcpy(path1,path); 
for (int i=0;i<lz;i++){
     for (int k=0;k<ly*lx;k++) fread(&xytmph[k],sizeof(xytmph[k]),1,file);  
     cudaMemcpy(xytmp,xytmph, ly*lx*sizeof(float), cudaMemcpyHostToDevice);
     setXYplane<<<grid3D,blocks3D>>>(xytmp,lx,ly,lz,i,0); } fclose(file);
//read x-velocity
FILE* file1 = fopen (strcat(path1,"ux3Dii"), "rb"); 
printf("file1: %s\n", path1); strcpy(path1,path);
for (int i=0;i<lz;i++){
     for (int k=0;k<ly*lx;k++) fread(&xytmph[k],sizeof(xytmph[k]),1,file1); 
     cudaMemcpy(xytmp,xytmph, ly*lx*sizeof(float), cudaMemcpyHostToDevice);
     setXYplane<<<grid3D,blocks3D>>>(xytmp,lx,ly,lz,i,1); } fclose(file1);
//read y-velocity
FILE* file2 = fopen (strcat(path1,"uy3Dii"), "rb"); 
printf("file2: %s\n", path1); strcpy(path1,path);
for (int i=0;i<lz;i++){
     for (int k=0;k<ly*lx;k++) fread(&xytmph[k],sizeof(xytmph[k]),1,file2); 
     cudaMemcpy(xytmp,xytmph, ly*lx*sizeof(float), cudaMemcpyHostToDevice);
     setXYplane<<<grid3D,blocks3D>>>(xytmp,lx,ly,lz,i,2); } fclose(file2);
//read z-velocity
FILE* file3 = fopen (strcat(path1,"uz3Dii"), "rb"); 
printf("file3: %s\n", path1); strcpy(path1,path);
for (int i=0;i<lz;i++){
     for (int k=0;k<ly*lx;k++) fread(&xytmph[k],sizeof(xytmph[k]),1,file3); 
     cudaMemcpy(xytmp,xytmph, ly*lx*sizeof(float), cudaMemcpyHostToDevice);
     setXYplane<<<grid3D,blocks3D>>>(xytmp,lx,ly,lz,i,3); } fclose(file3);

//read the time dependent (monitored) data from qpii
file  = fopen (strcat(path1,"qpii.txt"), "rt");  strcpy(path1,path);         
file1  = fopen (strcat(path1,"qp.txt"), "wt");  strcpy(path1,path);         

char ch = fgetc(file); while (ch != EOF){ fputc(ch, file1); ch = fgetc(file); }  // restore qp.txt
fclose(file); fclose(file1);
printf("\n Velocity-pressure data has been read for continuing the calculations.");
} 
//read the previously saved average field and turbulence statistical quantities (see their definition in the main program)
 void initav(double *uxav, double *uyav, double *uzav, double *ux2av, double *uy2av, 
         double *uz2av, double *pav, double *p2av, double *uxyav, double *uxzav,double *uyzav,
                 int &cnt, int &cnt2, int lx,int ly,int lz, char *path ){
  char path1[pathlen];  strcpy(path1,path);
  // open average pressure and velocity files
    FILE* file  = fopen (strcat(path1,"_pii.txt"), "r");  strcpy(path1,path);       
    FILE* file1 = fopen (strcat(path1,"_uxii.txt"), "r"); strcpy(path1,path); 
    FILE* file2 = fopen (strcat(path1,"_uyii.txt"), "r"); strcpy(path1,path);
    FILE* file3 = fopen (strcat(path1,"_uzii.txt"), "r"); strcpy(path1,path);
  // open rms pressure and velocity files
    FILE* file4 = fopen (strcat(path1,"_p2ii.txt"), "r"); strcpy(path1,path);
    FILE* file5 = fopen (strcat(path1,"_ux2ii.txt"), "r"); strcpy(path1,path);
    FILE* file6 = fopen (strcat(path1,"_uy2ii.txt"), "r"); strcpy(path1,path);
    FILE* file7 = fopen (strcat(path1,"_uz2ii.txt"), "r"); strcpy(path1,path);
  // open mixed velocitiy files
    FILE* file8  = fopen (strcat(path1,"_uxy2ii.txt"), "r"); strcpy(path1,path);
    FILE* file9  = fopen (strcat(path1,"_uxz2ii.txt"), "r");  strcpy(path1,path);
    FILE* file10 = fopen (strcat(path1,"_uyz2ii.txt"), "r"); strcpy(path1,path);
  // open&read number of averaged cycles for mean fields and mean statistics
    FILE* file11 = fopen (strcat(path1,"_avii.txt"), "r"); 
    fscanf(file11,"%d %d",&cnt,&cnt2); fclose(file11);
  // read averaged data
    float tmp;
 for (int k=0;k<ly*(lz);k++) {
    fscanf(file,"%e\n",&tmp);  pav[k]=double(tmp)*double(cnt);
    fscanf(file1,"%e\n",&tmp); uxav[k]=double(tmp)*double(cnt); 
    fscanf(file2,"%e\n",&tmp); uyav[k]=double(tmp)*double(cnt);
    fscanf(file3,"%e\n",&tmp); uzav[k]=double(tmp)*double(cnt);
    fscanf(file4,"%e\n",&tmp); p2av[k]=double(tmp)*double(cnt2);
    fscanf(file5,"%e\n",&tmp); ux2av[k]=double(tmp)*double(cnt2);
    fscanf(file6,"%e\n",&tmp); uy2av[k]=double(tmp)*double(cnt2);
    fscanf(file7,"%e\n",&tmp); uz2av[k]=double(tmp)*double(cnt2);
    fscanf(file8,"%e\n",&tmp); uxyav[k]=double(tmp)*double(cnt2);
    fscanf(file9,"%e\n",&tmp); uxzav[k]=double(tmp)*double(cnt2);
    fscanf(file10,"%e\n",&tmp);uyzav[k]=double(tmp)*double(cnt2);  }

fclose(file);fclose(file1);fclose(file2);fclose(file3);fclose(file4);
fclose(file5);fclose(file6);fclose(file7);fclose(file8);fclose(file9);
fclose(file10);
}

//kernel for initializing the input vector to 0
__global__ void set_v( float *vec, unsigned long vecLen){
unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
if ((idx<vecLen)) {vec[idx]= 0.0;}}

//read the geometry from the textfile "pipe.txt" 
/* (can be generated by running the file pipe.m under MATLAB or Octave)
*/
void read_geom (char * hIO){
  FILE* file = fopen ("pipe.txt", "r");
  int i = 0; long k=0;  
  fscanf (file, "%d", &i);  hIO[k]=char(i); k=k+1;
  while (!feof (file)) {
      fscanf (file, "%d", &i);
      hIO[k]=char(i); k=k+1; }
  fclose (file);}

// ************************ start of the main program ***********************

int main(void){
    cudaSetDevice(0); 
    // utility vectors for acessing velocity and pressure components of the 3D textures
    float *test, *testh, *testh1, *yztmp,*yztmph,*yztmph1;
    // IO and hIO are the mask 
    char *IO, *hIO, *IOt1, *hIOt, path[pathlen]; 
    // parameters read from "param.txt", the file which contains the run-parameters
    float nu, pin, q1, q2, p1, p2, l1, l2, dtddy, tblend, blend;   
    int d1= 2, delta=0;
    // read parameters defined in param.txt
    FILE* file = fopen ("param.txt", "r");
// the avlues below are numerical inputs to the lattice Boltzmann method
// for computing their physical equivalent, rescaling is performed within the MATLAB scipts
       fscanf(file,"%e\n",&tblend); // timestep = dx*tblend
       fscanf(file,"%e\n",&pin);    // inlet pressure
       fscanf(file,"%e\n",&nu);     // numerical viscosity       
       fscanf(file,"%d\n",&d1);     // coarse corrective stencil length   
       fscanf(file,"%d\n",&delta);  // type of averaging switch
       fscanf(file,"%e\n",&blend);  // blend for the corrective discretization order
       while (fgets(path, pathlen, file) != NULL); // path for saved files/backup       
               sprintf(path,"");
       fclose(file); 
  // set the dimensions for the pipe  
  int lx=100; int ly=100; int lz=300; int r=(lx)/2;
  // add some wall sections and sum buffer section for the periodic boundary conditions
  int d2=10; lx+=2*d2; ly+=2*d2; lz+=2*d2;
  // number of timestep cycles 
  long cycle =500000000;
  
  // allocate mask variable on the host CPU and on the GPU, initialize host values to 0
  hIO =   (char*)malloc(lx*ly*(lz)*sizeof(char)); hIOt= (char*)malloc(ly*(lz)*sizeof(char));
  cudaMalloc(&IO,lx*ly*lz*sizeof(char));   cudaMalloc(&IOt1, ly*(lz)*sizeof(char));
  for (long i=0;i<lx*ly*(lz);i++) hIO[i]=0; for (long i=0;i<ly*(lz);i++) hIOt[i]=0;
  
  // allocate and initialize utility vectors on the host CPU for velcity and /or pressure processing 
  testh = (float*)malloc(lx*ly*sizeof(float));
  testh1 = (float*)malloc(lx*ly*sizeof(float));
  for (int i=0;i<lx*ly;i++) {testh[i]=0.0f;testh1[i]=0.0f;}
  yztmph = (float*)malloc(ly*(lz)*sizeof(float));
  for (int i=0;i<ly*(lz);i++) yztmph[i]=0.0f;
  yztmph1 = (float*)malloc(ly*(lz)*sizeof(float));
  for (int i=0;i<ly*(lz);i++) yztmph1[i]=0.0f;

  // allocate utility vector on the GPU for velocity and/or pressure processing 
  cudaMalloc(&test, lx*ly*sizeof(float));
  cudaMalloc(&yztmp, ly*lz*sizeof(float));
 
  // create 3D CUDA-array for texture storage of the velocity and pressure
  cudaExtent vol = {size_t(lx),size_t(ly),size_t(lz)};
  
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
  cudaArray_t H;
  cudaMalloc3DArray(&H, &channelDesc, vol, cudaArraySurfaceLoadStore);
  cudaBindSurfaceToArray(surf,H);
  cudaBindTextureToArray(vtexr, H);    // bind array to 3D texture
 
// define and allocate average quantities, and their related variables 
/*(they will contain field averages and turbuelence statistics)
*/
    int cnt=0; int cnt2=0;    
    double *uxav,*uyav,*uzav,*ux2av,*uy2av,*uz2av,*pav,*p2av,*uxyav,*uxzav,*uyzav;
    // x- velocity average in a selected yz plane
    uxav = (double*)malloc(ly*(lz)*sizeof(double));
    // y- velocity average in a selected yz plane
    uyav = (double*)malloc(ly*(lz)*sizeof(double));
    // z- velocity average in a selected yz plane
    uzav = (double*)malloc(ly*(lz)*sizeof(double));
    //    pressure average in a selected yz plane
     pav = (double*)malloc(ly*(lz)*sizeof(double));

    // rms - root-mean-square velocity in the x-direction
   ux2av = (double*)malloc(ly*(lz)*sizeof(double));
    // rms y-velocity
   uy2av = (double*)malloc(ly*(lz)*sizeof(double));
    // rms z-velocity 
   uz2av = (double*)malloc(ly*(lz)*sizeof(double));
    // rms pressure
    p2av = (double*)malloc(ly*(lz)*sizeof(double));

    // rms xy velocity
   uxyav = (double*)malloc(ly*(lz)*sizeof(double));
    // rms xz velocity
   uxzav = (double*)malloc(ly*(lz)*sizeof(double));
    // rms yz velocity
   uyzav = (double*)malloc(ly*(lz)*sizeof(double));

  // initialize averages quantitites to 0
    for (int i=0;i<ly*(lz);i++) {
   // quantities that contain average velocity and pressure in a pre-selected plane      
   uxav[i] = 0.0; uyav[i] = 0.0;uzav[i] = 0.0; pav[i] = 0.0;
   // quantities that contain average RMS velocity and RMS pressure in a pre-selected plane      
   ux2av[i] = 0.0;uy2av[i] = 0.0; uz2av[i]=0.0; p2av[i]=0.0;
   // quantities that contain average mixed velocity in a pre-selected plane      
    uxyav[i]=0.0;uxzav[i]=0.0;uyzav[i]=0.0;}
//definition finished 
// read mask containing fluid region, wall and inlet
  read_geom(hIO);   	                          
// counters that contain number of cell elements at the inlet and outlet          
  l1=0.0f; l2=0.0f; 
// find number of inlet and outlet cells
  for (unsigned long i=0;i<lx*ly*(lz);i++) 
        if(hIO[i]==1) { l1=l1+1.0f/float(2*d2);} 
  l2=l1;
// copy the mask to device memory
  cudaMemcpy(IO,hIO, lx*ly*(lz)*sizeof(char), cudaMemcpyHostToDevice);    
  free(hIO);
// open a file that will contain various quantities monitored during the run at the inlet and outlet
// e.g. average velocity, pressure, friction factors, Re, etc.        
  FILE* fileqp=fopen("qp.txt","a");  fclose(fileqp);
// setup numerics for LBM        
  dtddy=(nu-0.16666666f); 
//setup grids and blocks for the CUDA kernels
  dim3 blocks2D(8,8); 
  dim3 grid2D(lx/8+1,ly/8+1);
  dim3 blocks3D(8,8,8); 
  dim3 grid3D(lx/8+1,ly/8+1,lz/8+1);
  dim3 blocks3D1(bdx,bdy,bdz); 
  dim3 grid3D1(lx/bdx+1,ly/bdy+1,1);
   //initialize velocity and pressure (initphys) or read them from previous run (initphys1, initav)
   // initialize to [p,ux,uy,uz]=[0,0,0,0.0f]
//   initphys<<<grid3D,blocks3D>>>(pin,lx,ly,lz,d2);    
   initphys1(test, testh, lx, ly, lz, path); 
   initav(uxav,uyav,uzav,ux2av,uy2av,uz2av,pav,p2av,uxyav,uxzav,uyzav,cnt,cnt2,lx,ly,lz, path);

//set utility vectors to 0
  set_v<<< lx*ly/64, 64 >>>(test, lx*ly);
  set_v<<< lx*(lz)/64, 64 >>>(yztmp, ly*(lz));
   float Re;   
//   unsigned long i; 
         double L=0.0; int k=0;
//start the calculation 
  for (unsigned long c=1;c<cycle;c++){
//scan through blocks of 8-XY planes and update by shifting the results 
  for ( k = (lz-1) - d2; k >= d2; k -= bdz){
  FSLBMcalc<<< grid3D1, blocks3D1 >>>(IO, lx, ly, lz, dtddy, pin, nu, d2, k, 
          tblend, d1,1.0/float(d1*d1),blend); } // LBM kernel
//after a full update shift the results in their place        
  for (int k=0;k<(lz-d2);k++)  shiftback<<<grid2D,blocks2D>>>(lx,ly,lz,d2,k);
//apply periodic boundary conditions
  bcond<<<grid3D,blocks3D>>>(IO,lx,ly,lz,d2,pin,p2);  p2=0.0f;
// 50 timestep cylcles output
if ((c%50)==0){ 
    //initialize average velocity and pressure at the inlet and outlet
    q1=0.0f; q2=0.0f; p1=0.0f; p2=0.0f;  
	// physical variables		        
        //get outlet quantitites of interest and sum them
		getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,lz-d2,3);  
        cudaMemcpy(testh,test, lx*ly*sizeof(float), cudaMemcpyDeviceToHost); 
		getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,lz-d2,0); 
        cudaMemcpy(testh1,test, lx*ly*sizeof(float), cudaMemcpyDeviceToHost); 
 		for(int i=0;i<lx*ly;i++) {  {p2+=testh1[i];q2+=testh[i];}  }
        //get inlet quantitites of interest and sum them
		getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,d2,3);  
        cudaMemcpy(testh,test, lx*ly*sizeof(float), cudaMemcpyDeviceToHost); 
		getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,d2,0); 
        cudaMemcpy(testh1,test, lx*ly*sizeof(float), cudaMemcpyDeviceToHost); 
     	        for(int i=0;i<lx*ly;i++) {  {p1+=testh1[i];q1+=testh[i];}  }	   
        // calculate their average
        p1=p1/l1; p2=p2/l2; q1=q1/l1; q2=q2/l2;

        // calculate Reynold number
        Re=float(2*r)*q1/nu;
        
        //get a plane of interest for output of z-velocity oscillation at (x0,y0,z0)=(lx/2,ly/2,lz/2)
		getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,lz/2,3);  
        cudaMemcpy(testh,test, lx*ly*sizeof(float), cudaMemcpyDeviceToHost); 
/* output the timeste, the numerical viscosity, average z-velocity at inlet/outlet,
   average pressure at inlet/outlet, friction factor, the sample point defined 
   above, Re, number of inlet cells and number of outlet cells
*/      printf("time step: %d, dtddy: %e, q1=%e, q2=%e, p1 =%e, p2=%e, f.f.=%e, \
Re=%e, #inl. elements=%d, #outl. elements=%d\n",c,dtddy,q1,q2,p1,p2,\
        (p1-p2)/q1/q1/3.0f*4.0f*float(r)/float(lz-2*d2),Re,int(l1),int(l2));        
/*         store the average z-velocity at inlet/outlet, average pressure at inlet/outlet, 
                friction factor, and the sample point defined above
*/     FILE* fileqp=fopen("qp.txt","a");  
       fprintf(fileqp,"%e %e %e %e %e %e\n",q1,q2,p1,p2,(p1-p2)/q1/q1/3.0f*4.0f*float(r)/float(lz-2*d2),
               testh[ly*lx/2+lx/2],L); 
       fclose(fileqp);   
        if ((c%500)==0) {
        // read the parameter file durin run if any changes are needed     
       file = fopen ("param.txt", "r"); 
       fscanf(file,"%e\n",&tblend); 
       fscanf(file,"%e\n",&pin); 
       fscanf(file,"%e\n",&nu); 
       fscanf(file,"%d\n",&d1);
       fscanf(file,"%d\n",&delta);  fscanf(file,"%e\n",&blend); //fscanf(file,"%s",path);
       while (fgets(path, pathlen, file) != NULL);   
       fclose(file);   
       dtddy=(nu-0.16666666f);      
       printf("\nParameter file has been read.\n\n"); 
  
       // write average fields and turbulence statistics       
 	   printf("calculating/writing avg. p-u files, avg1(%d), avg2(%d)",cnt,cnt2);       
       // reset average fields and turbulence statistics for delta=0
        if (delta==0) {cnt=0; cnt2=0; for (int i=0;i<ly*(lz);i++) {
           uxav[i] = 0.0; uyav[i] = 0.0; uzav[i] = 0.0; pav[i] = 0.0; ux2av[i] = 0.0;
           uy2av[i] = 0.0; uz2av[i] = 0.0; p2av[i] = 0.0; uxyav[i] = 0.0;
           uxzav[i] = 0.0; uyzav[i] = 0.0;}}
       // restet turbulence statistics but not average fields for delta=1          
        if (delta==1) {cnt2=0; for (int i=0;i<ly*(lz);i++) {
            ux2av[i] = 0.0; uy2av[i] = 0.0; uz2av[i] = 0.0; p2av[i] = 0.0;
            uxyav[i] = 0.0; uxzav[i] = 0.0; uyzav[i] = 0.0;}}       

    FILE* file = fopen ("_p.txt", "wt");    // average pressure
    FILE* file1 = fopen ("_ux.txt", "wt");  // average x-velocity
    FILE* file2 = fopen ("_uy.txt", "wt"); 
    FILE* file3 = fopen ("_uz.txt", "wt");

    FILE* file4 = fopen ("_p2.txt", "wt");  // average RMS pressure 
    FILE* file5 = fopen ("_ux2.txt", "wt"); // average RMS x-velocity
    FILE* file6 = fopen ("_uy2.txt", "wt");
    FILE* file7 = fopen ("_uz2.txt", "wt");

    FILE* file8  = fopen ("_uxy2.txt", "wt");  // average x-y velocity correlation
    FILE* file9  = fopen ("_uxz2.txt", "wt"); 
    FILE* file10 = fopen ("_uyz2.txt", "wt");
    FILE* file11 = fopen ("_av.txt","wt");    //file containing th number of field and statistics averages performed
    // turbulent pressure fluctuation
    double *dd1= (double*)malloc(ly*(lz)*sizeof(double));
     // turbulent x-velocity fluctuation
    double *dd2= (double*)malloc(ly*(lz)*sizeof(double));
    // turbulent y-velocity fluctuation
    double *dd3= (double*)malloc(ly*(lz)*sizeof(double));
    // turbulent z-velocity fluctuation
    double *dd4= (double*)malloc(ly*(lz)*sizeof(double));

      cnt=cnt+1; cnt2=cnt2+1;

      getYZplane<<<grid3D,blocks3D>>>(yztmp,lx,ly,lz,lx/2,0); 
      cudaMemcpy(yztmph,yztmp, ly*(lz)*sizeof(float), cudaMemcpyDeviceToHost); 
      for (int k=0;k<ly*(lz);k++){
       pav[k]=pav[k]+double(yztmph[k]);               // average pressure
       dd1[k]=pav[k]/double(cnt)-double(yztmph[k]);}  // turbulent pressure fluctutation

      getYZplane<<<grid3D,blocks3D>>>(yztmp,lx,ly,lz,lx/2,1); 
      cudaMemcpy(yztmph,yztmp, ly*(lz)*sizeof(float), cudaMemcpyDeviceToHost); 
      for (int k=0;k<ly*(lz);k++){
       uxav[k]=uxav[k]+double(yztmph[k]);            // average x-velocity
       dd2[k]=uxav[k]/double(cnt)-double(yztmph[k]);}//turbulent x-velocity fluctuation

      getYZplane<<<grid3D,blocks3D>>>(yztmp,lx,ly,lz,lx/2,2); 
      cudaMemcpy(yztmph,yztmp, ly*(lz)*sizeof(float), cudaMemcpyDeviceToHost); 
      for (int k=0;k<ly*(lz);k++){
       uyav[k]=uyav[k]+double(yztmph[k]); 
       dd3[k]=uyav[k]/double(cnt)-double(yztmph[k]);}//turbulent y-velocity fluctuation

      getYZplane<<<grid3D,blocks3D>>>(yztmp,lx,ly,lz,lx/2,3); 
      cudaMemcpy(yztmph,yztmp, ly*(lz)*sizeof(float), cudaMemcpyDeviceToHost); 
      for (int k=0;k<ly*(lz);k++){
       uzav[k]=uzav[k]+double(yztmph[k]); 
       dd4[k]=uzav[k]/double(cnt)-double(yztmph[k]);}//turbulent z-velocity fluctuation
     // calculate correlations for RMS pressure and Reynolds stress tensor
      for (int k=0;k<ly*(lz);k++){
       p2av[k]=p2av[k]  +abs(dd1[k]);
       ux2av[k]=ux2av[k]+abs(dd2[k]);
       uy2av[k]=uy2av[k]+abs(dd3[k]);
       uz2av[k]=uz2av[k]+abs(dd4[k]);
       uxyav[k]=uxyav[k]+dd2[k]*dd3[k]; 
       uxzav[k]=uxzav[k]+dd2[k]*dd4[k]; 
       uyzav[k]=uyzav[k]+dd4[k]*dd3[k];}

    free(dd1);free(dd2);free(dd3);free(dd4);
    // write correlations
    for (int k=0;k<ly*(lz);k++) {
    fprintf(file,"%e\n",pav[k]/double(cnt));      // avg. p
    fprintf(file1,"%e\n",uxav[k]/double(cnt));    // avg. ux
    fprintf(file2,"%e\n",uyav[k]/double(cnt));    // avg. uy
    fprintf(file3,"%e\n",uzav[k]/double(cnt));    // avg. uz
    fprintf(file4,"%e\n",p2av[k]/double(cnt2));   // avg. (avg. p-p)^2 }
    fprintf(file5,"%e\n",ux2av[k]/double(cnt2));  // avg. (avg. ux-ux)^2
    fprintf(file6,"%e\n",uy2av[k]/double(cnt2));  // avg. (avg. uy-uy)^2
    fprintf(file7,"%e\n",uz2av[k]/double(cnt2));  // avg. (avg. uz-uz)^2
    fprintf(file8,"%e\n",uxyav[k]/double(cnt2));  // avg. (avg. ux-ux)(avg. uz-uz)
    fprintf(file9,"%e\n",uxzav[k]/double(cnt2));  // avg. (avg. ux-ux)(avg. uz-uz)
    fprintf(file10,"%e\n",uyzav[k]/double(cnt2));}// avg. (avg. uy-uy)(avg. uz-uz)
    fprintf(file11,"%d %d\n",cnt,cnt2);

    fclose(file); fclose(file1); fclose(file2); fclose(file3); 
    fclose(file4); fclose(file5); fclose(file6); fclose(file7); 
    fclose(file8); fclose(file9); fclose(file10); fclose(file11); 
    printf(" ... p-u files written\n");//} 

    // output the predefined fields for the current time step
	printf("writing p-u files");       // output the p-v data in a plane
     file = fopen ("p.txt", "wt");     // pressure in the yz plane at x=lx/2
    file1 = fopen ("ux.txt", "wt");    // x-velocity in the yz plane at x=lx/2
    file2 = fopen ("uy.txt", "wt");    // y-velocity in the yz plane at x=lx/2 
    file3 = fopen ("uz.txt", "wt");    // z-velocity in the yz plane at x=lx/2 
    file4 = fopen ("IO.txt", "wt");    // mask in the yz plane at x=lx/2
    
	file6 = fopen ("ux1.txt", "wt");   // x-velocity in the XY plane at z=lz/2
    file7 = fopen ("uy1.txt", "wt");   // y-velocity in the XY plane at z=lz/2
    file8 = fopen ("uz1.txt", "wt");   // z-velocity in the XY plane at z=lz/2
    file9 = fopen ("p1.txt", "wt");    //   pressure in the XY plane at z=lz/2
    // get and write YZ pressure at x=lx/2
	getYZplane<<<grid3D,blocks3D>>>(yztmp,lx,ly,lz,lx/2,0); 
    cudaMemcpy(yztmph,yztmp, ly*(lz)*sizeof(float), cudaMemcpyDeviceToHost);  
    for (int k=0;k<ly*(lz);k++) fprintf(file,"%e\n",yztmph[k]);
    // get and write YZ x-velocity at x=lx/2
	getYZplane<<<grid3D,blocks3D>>>(yztmp,lx,ly,lz,lx/2,1); 
    cudaMemcpy(yztmph,yztmp, ly*(lz)*sizeof(float), cudaMemcpyDeviceToHost);  
    for (int k=0;k<ly*(lz);k++) fprintf(file1,"%e\n",yztmph[k]);
    // get and write YZ y-velocity at x=lx/2
	getYZplane<<<grid3D,blocks3D>>>(yztmp,lx,ly,lz,lx/2,2); 
    cudaMemcpy(yztmph,yztmp, ly*(lz)*sizeof(float), cudaMemcpyDeviceToHost);  
    for (int k=0;k<ly*(lz);k++) fprintf(file2,"%e\n",yztmph[k]); 
    // get and write YZ z-velocity at x=lx/2
	getYZplane<<<grid3D,blocks3D>>>(yztmp,lx,ly,lz,lx/2,3); 
    cudaMemcpy(yztmph,yztmp, ly*(lz)*sizeof(float), cudaMemcpyDeviceToHost);  
    for (int k=0;k<ly*(lz);k++) fprintf(file3,"%e\n",yztmph[k]);  
    // get and write XY x-velocity at z=lz/2
  	getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,lz/2,1);  
    cudaMemcpy(testh,test, lx*ly*sizeof(float), cudaMemcpyDeviceToHost); 
    for (int k=0;k<ly*lx;k++) fprintf(file6,"%e\n",testh[k]); 
    // get and write XY y-velocity at z=lz/2
  	getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,lz/2,2);  
    cudaMemcpy(testh,test, lx*ly*sizeof(float), cudaMemcpyDeviceToHost); 
    for (int k=0;k<ly*lx;k++) fprintf(file7,"%e\n",testh[k]);
    // get and write XY z-velocity at z=lz/2
  	getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,lz/2,3);  
    cudaMemcpy(testh,test, lx*ly*sizeof(float), cudaMemcpyDeviceToHost); 
    for (int k=0;k<ly*lx;k++) fprintf(file8,"%e\n",testh[k]);
    // get and write XY pressure at z=lz/2
  	getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,lz/2,0);  
    cudaMemcpy(testh,test, lx*ly*sizeof(float), cudaMemcpyDeviceToHost); 
    for (int k=0;k<ly*lx;k++) fprintf(file9,"%e\n",testh[k]);
    // get and write YZ mask at x=lx/2
	getYZplaneIO<<<lx*ly*(lz)/128,128>>>(IO,IOt1,lx,ly,lz,lx/2); 
    cudaMemcpy(hIOt,IOt1, ly*(lz)*sizeof(char), cudaMemcpyDeviceToHost);  
    for (int k=0;k<ly*(lz);k++) fprintf(file4,"%d\n",hIOt[k]);	
    fclose(file); fclose(file1); fclose(file2); fclose(file3); fclose(file4); 
    fclose(file6); fclose(file7); fclose(file8); fclose(file9);
    printf(" ... p-u files written\n");

   }}
// full backup at 50000 timesteps
if ((c%50000)==0) {
 	printf("performing backup-save for the run in progress ...");  
        char path1[pathlen]; strcpy(path1,path);
     // pressure
             
     FILE* file = fopen (strcat(path1,"p3Dii"), "wb");  strcpy(path1,path);
     // x-velocity
	FILE* file1 = fopen (strcat(path1,"ux3Dii"), "wb"); strcpy(path1,path);
     // y-velocity
	FILE* file2 = fopen (strcat(path1,"uy3Dii"), "wb"); strcpy(path1,path);
     // z-velocity
	FILE* file3 = fopen (strcat(path1,"uz3Dii"), "wb"); strcpy(path1,path);

     // write full 3D data files
	for (int i=0;i<lz;i++){
    //    pressure
	getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,i,0);  
    cudaMemcpy(testh,test, ly*lx*sizeof(float), cudaMemcpyDeviceToHost);  
    for (int k=0;k<ly*lx;k++) fwrite(&testh[k],sizeof(float),1,file); 
    //    x-velocity
	getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,i,1);  
    cudaMemcpy(testh,test, ly*lx*sizeof(float), cudaMemcpyDeviceToHost);  
    for (int k=0;k<ly*lx;k++) fwrite(&testh[k],sizeof(float),1,file1);
    //    y-velocity
	getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,i,2);  
    cudaMemcpy(testh,test, ly*lx*sizeof(float), cudaMemcpyDeviceToHost);  
    for (int k=0;k<ly*lx;k++) fwrite(&testh[k],sizeof(float),1,file2);// fprintf(file2,"%e ",xytmph[k]); 
            
	getXYplane<<<grid3D,blocks3D>>>(test,lx,ly,lz,i,3);  
    cudaMemcpy(testh,test, ly*lx*sizeof(float), cudaMemcpyDeviceToHost);  
    for (int k=0;k<ly*lx;k++) fwrite(&testh[k],sizeof(float),1,file3);// fprintf(file3,"%e ",xytmph[k]);  
	}
	fclose(file); fclose(file1); fclose(file2); fclose(file3);

    // backup for the average quantities
    char ch;
           file = fopen (strcat(path1,"qp.txt"), "rt");  strcpy(path1,path);
           file1 = fopen (strcat(path1,"qpii.txt"), "wt");  strcpy(path1,path);

    ch = fgetc(file); while (ch != EOF){ fputc(ch, file1); ch = fgetc(file); }  // backup qp.txt
    fclose(file); fclose(file1);
   // average pressure
            file = fopen (strcat(path1,"_pii.txt"), "wt");  strcpy(path1,path);
  // average  velocity
           file1 = fopen (strcat(path1,"_uxii.txt"), "wt");  strcpy(path1,path);
           file2 = fopen (strcat(path1,"_uyii.txt"), "wt");  strcpy(path1,path);
           file3 = fopen (strcat(path1,"_uzii.txt"), "wt"); strcpy(path1,path);
  // average rms pressure
     FILE* file4 = fopen (strcat(path1,"_p2ii.txt"), "wt"); strcpy(path1,path);
 // average rms x-velocity
     FILE* file5 = fopen (strcat(path1,"_ux2ii.txt"), "wt"); strcpy(path1,path); 
// average rms y-velocity
     FILE* file6 = fopen (strcat(path1,"_uy2ii.txt"), "wt"); strcpy(path1,path);
     FILE* file7 = fopen (strcat(path1,"_uz2ii.txt"), "wt"); strcpy(path1,path);
// average rms xy correlation
    FILE* file8  = fopen (strcat(path1,"_uxy2ii.txt"), "wt"); strcpy(path1,path); 
    FILE* file9  = fopen (strcat(path1,"_uxz2ii.txt"), "wt");  strcpy(path1,path);
    FILE* file10 = fopen (strcat(path1,"_uyz2ii.txt"), "wt"); strcpy(path1,path);
// file containing the number of timesteps that have been averaged
    FILE* file11 = fopen (strcat(path1,"_avii.txt"),"wt");                        
    // write the averaged fields and statistics
    for (int k=0;k<ly*(lz);k++) {
    fprintf( file,"%e\n",pav[k]/double(cnt));      
    fprintf(file1,"%e\n",uxav[k]/double(cnt)); 
    fprintf(file2,"%e\n",uyav[k]/double(cnt)); 
    fprintf(file3,"%e\n",uzav[k]/double(cnt));
    fprintf(file4,"%e\n",p2av[k]/double(cnt2));   
    fprintf(file5,"%e\n",ux2av[k]/double(cnt2)); 
    fprintf(file6,"%e\n",uy2av[k]/double(cnt2)); 
    fprintf(file7,"%e\n",uz2av[k]/double(cnt2));
    fprintf(file8,"%e\n",uxyav[k]/double(cnt2)); 
    fprintf(file9,"%e\n",uxzav[k]/double(cnt2));
    fprintf(file10,"%e\n",uyzav[k]/double(cnt2));}
    // write the number of averaged timesteps for the fields and for the turbulence statistics
    fprintf(file11,"%d %d\n",cnt,cnt2);

    fclose(file); fclose(file1); fclose(file2); fclose(file3); fclose(file4);
   fclose(file5); fclose(file6); fclose(file7); fclose(file8); fclose(file9);
   fclose(file10); fclose(file11); 
 	printf(" ... backup save finished\n"); }
  }  
/* MATLAB syntax for saving geometry
IO=uint8(IO); fid = fopen('mediumfine.txt', 'wt'); fprintf(fid, '%d\n', IO); fclose(fid)
*/  
}
