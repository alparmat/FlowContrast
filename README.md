#SharpFlow
CUDA source for fractional step lattice Boltzmann methods i.e. by sharpening of fluid flow with reverse diffusion

This is a demo showing the initial idea (see documentation/paper file: MethodDoc_withFSLBMpaper.pdf)
Running the program:
use MATLAB or Octave to execute pipe.m which generates the domain for the calculations.
Build FSLBM.cu with nvcc into an executable
run the executable
use the rest of the .m files to visualize the flow
note:
  If you run the demo with a changed domain size you will have to change also the size values in the .cu file
  If you change the domain geometry you either need to implement new boundary conditions or change the periodic ones to fit
  
  next benchmark to follow: Taylor-Green vortex
