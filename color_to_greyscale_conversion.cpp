#include<hip/hip_runtime.h>
#include<opencv2/opencv.hpp>
__global__ void colorToGreyscaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int
height) {
	int Col = threadIdx.x + blockIdx.x * blockDim.x;
	int Row = threadIdx.y + blockIdx.y * blockDim.y;
	if (Col < width && Row < height) {
		int greyOffset = Row * width + Col;
		int rgbOffset = greyOffset * 3;
		unsigned char r = Pin[rgbOffset]; // Red
		unsigned char g = Pin[rgbOffset + 1]; // Green
		unsigned char b = Pin[rgbOffset + 2]; // Blue
		Pout[greyOffset] = 0.21f * r + 0.72f * g + 0.07f * b;
	}
}
// Host code
int main() {
// Image dimensions
int width = 512; // Example width
int height = 512; // Example height
int numPixels = width * height;
float gpu_elapsed_time_ms;
// Generate a simple synthetic color image using OpenCV (just for visualization)
cv::Mat h_colorImg(height, width, CV_8UC3);
for (int i = 0; i < height; ++i) {
	for (int j = 0; j < width; ++j) {
		h_colorImg.at<cv::Vec3b>(i, j) = cv::Vec3b(j % 256, i % 256, (i + j) % 256); 
	}
}
// Save the original color image for comparison
cv::imwrite("color_image.png", h_colorImg);
// Allocate host memory for the output image
unsigned char *h_Pout = new unsigned char[numPixels]; // Greyscale image
// Allocate device memory
unsigned char *d_Pin, *d_Pout;
hipMalloc(&d_Pin, numPixels * 3 * sizeof(unsigned char));
hipMalloc(&d_Pout, numPixels * sizeof(unsigned char));
// Copy input data from host to device
hipMemcpy(d_Pin, h_colorImg.data, numPixels * 3 * sizeof(unsigned char),hipMemcpyHostToDevice);
// Define grid and block dimensions
dim3 dimBlock(16, 16, 1);
dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) /dimBlock.y, 1);
hipEvent_t start,stop;
hipEventCreate(&start);
hipEventCreate(&stop);
hipEventRecord(start,0);
// Launch the kernel
colorToGreyscaleConversion<<<dimGrid, dimBlock>>>(d_Pout, d_Pin, width, height);
hipEventRecord(stop,0);
hipEventSynchronize(stop);
hipMemcpy(h_Pout, d_Pout, numPixels * sizeof(unsigned char), hipMemcpyDeviceToHost);
// Convert the output array to an OpenCV matrix and save the result
hipDeviceSynchronize();
hipEventElapsedTime(&gpu_elapsed_time_ms,start,stop);
printf("\nTime elapsed on processing colortogreyscale conversion  on GPU:%f ms.\n\n",gpu_elapsed_time_ms);
cv::Mat h_greyscaleImg(height, width, CV_8UC1, h_Pout);
cv::imwrite("greyscale_image.png", h_greyscaleImg);
// Free device memory
hipFree(d_Pin);
hipFree(d_Pout);
delete[] h_Pout;
std::cout << "Color to greyscale conversion completed successfully!" << std::endl;
std::cout << "Images saved as color_image.png and greyscale_image.png" << std::endl;
return 0;
}
								      

