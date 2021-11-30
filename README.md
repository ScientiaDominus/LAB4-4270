# LAB4-4270
A CUDA image processing lab written in C and C++ using the CUDA library.

# Running the Program 
This program uses commandline file input to specify the image file you would like to process. 
An important note to make is that this program only processes Portable Gray Map (PGM) format images. 
in order to run this program first you must load the cuda8 module. Then use nvcc to compile the img_prcs.cu file. 
Once this is complete you can launched the program like so 

    ./a.out <file name here>.pgm

This will specify the file that you wish to use and send that to the program for processing. Once the program is
finished running there should be two files, "output_box.pgm" and "output_gauss.pgm" these are the processed results 
of the original image. These can be opened on any mainstream operating system by simply double clicking on the file. 
no special software is required. 


