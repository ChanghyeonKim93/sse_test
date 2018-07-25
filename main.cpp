#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <unistd.h>

#include <ctime>

//#include <ia32intrin.h>
//#include <xmmintrin.h> // sse
//#include <emmintrin.h> // sse2
//#include <smmintrin.h> // sse4
//#include <nmmintrin.h> // sse4
#include <immintrin.h> // Intel core SIMD intrinsic instructions
#include <pmmintrin.h> // sse3

#define ALIGN __attribute__((__aligned__(16))) // Macro for aligned memory
static const __m128 ZEROS = _mm_setzero_ps();
static const __m128 ONES  = _mm_set1_ps(1.0f);
static const __m128 NANS  = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());

float** allocate_float_2d(const int& npoints, const int& ndims) {
	float** ptr = new float*[npoints];
	for(size_t i = 0; i < npoints; i++) ptr[i] = new ALIGN float[ndims]();
	return ptr;
}
void delete_float_2d(float** ptr, const int& npoints) {
	for(size_t i = 0; i < npoints; i++) delete[] ptr[i];
	delete[] ptr;
	printf("float array (dynamic allocated) : deleted\n");
}

int main()
{
	printf("\n----------------------------\n");
	printf("-                          -\n");
	printf("-      program starts.     -\n");
	printf("-                          -\n");
	printf("----------------------------\n\n\n");

	clock_t start, finish;
	
	size_t ndims   = 4; 
	size_t npoints = 10000000;

	float** points_array     = allocate_float_2d(npoints, ndims);
	float** poitns_res_array = allocate_float_2d(npoints, ndims);
	float*  b = new float[npoints];
	for(size_t i = 0; i < npoints; i++)
	{
		b[i] = 100.0*(float)i+0.123456789123456789f;
		points_array[i][0] = b[i];
		points_array[i][1] = b[i];
		points_array[i][2] = b[i];
		points_array[i][3] = b[i];
	}


	float inv_depth;
	start = clock();
	for(size_t i = 0; i < npoints; i++)
	{
		inv_depth = 1.0f/b[i];
		//float inv_depth = b[i];
		poitns_res_array[i][0] = points_array[i][0] + inv_depth;
		poitns_res_array[i][1] = points_array[i][1] + inv_depth;
		poitns_res_array[i][2] = points_array[i][2] + inv_depth;
		poitns_res_array[i][3] = points_array[i][3] + inv_depth;
		// std::cout << poitns_res_array[i][0]<<","<<poitns_res_array[i][1]<<","<<poitns_res_array[i][2]<<","<<poitns_res_array[i][3] << std::endl;
	}
	finish = clock();
	std::cout <<" float time: "<< (finish - start) / 1000.0 << "[ms]" << std::endl;


	// SSE version
	__m128 pppp;
	__m128 dddd_duplicate;
	__m128 res;
	float res_float[4] = {0};
	dddd_duplicate = _mm_set1_ps(inv_depth);

	start = clock();
	for(size_t i = 0; i < npoints; i++)
	{	
		inv_depth = 1.0f/b[i];
		pppp           = _mm_load_ps(points_array[i]);
		dddd_duplicate = _mm_set1_ps(inv_depth);
		res            = _mm_add_ps(pppp, dddd_duplicate);
		//res = _mm_add_ps(_mm_load_ps(points_array[i]), dddd_duplicate);
		_mm_store_ps(res_float, res);
		//std::cout<<res[0]<<","<<res[1]<<","<<res[2]<<","<<res[3]<<std::endl;
		//std::cout<<res_float[0]<<","<<res_float[1]<<","<<res_float[2]<<","<<res_float[3]<<std::endl;
	}
	finish = clock();
	std::cout <<" sse time : " << (finish - start) / 1000.0 << "[ms]" << std::endl;

	delete_float_2d(points_array, npoints);
	std::cout << "End of programm." << std::endl;
	return 0;
}
