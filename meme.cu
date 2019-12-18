#include <cmath>
#include <string>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <chrono>
#define ll long long
#define ull unsigned long long
#define next(N) N*=25214903917;N+=11;N%=281474976710656
#define next3(N) N*=233752471717045; N+=11718085204285; N%=281474976710656//simulates 3 calls to next()
#define numBlocks 4096
#define numThreadsPerBlock 256
#define M_PI 3.1415926535897932

__device__ int nextInt94(ull seed);
__device__ bool goodChest(ull chunkseed);
__global__ void checkSeeds(ull* start, int* d_sinLUT, int* d_cosLUT, int* d_rad, int* d_srad);

int sinLUT[1024];
int cosLUT[1024];

void calculateLUTs(){ //Thanks to jacobsjo for adding a Look-Up Table
	for (int i = 0 ; i< 1024 ; i++){
		sinLUT[i] = round(sin((i* M_PI) / 512.0)*2048);
		cosLUT[i] = round(cos((i* M_PI) / 512.0)*2048);
	}
}

int main(int argc, char* argv[]) {
	using namespace std::chrono;

  cudaSetDevice(0);
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

	bool printTime = true;
	ull searchTo = 100000000000;
	ull searchFrom = 0;

	int rad = 64;
	int srad = 6;

	for (int i = 0; i < argc; i++) {
		std::string arg = std::string(argv[i]);
		if (std::string(argv[i]) == "-t") {
			std::string nextArg = std::string(argv[++i]);
			printTime = (nextArg != "false" && nextArg != "0");
		}
		else if (arg == "-s"){
			searchFrom = (ull)std::stoll (std::string(argv[++i]),NULL,0);
			searchTo = (ull)std::stoll (std::string(argv[++i]),NULL,0);
		}
		else if (arg == "-r") {
			rad = (int)std::stoi (std::string(argv[++i]),NULL,0);
		}
		else if (arg == "-sr") {
			srad = (int)std::stoi (std::string(argv[++i]),NULL,0);
		}
	}

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	int *d_sinLUT, *d_cosLUT;
	int *d_rad, *d_srad;
	ull *d_seed;
	int refresh = 0;

	calculateLUTs();

	cudaMalloc((void**)&d_sinLUT,sizeof(int)*1024);
	cudaMalloc((void**)&d_cosLUT,sizeof(int)*1024);
	cudaMalloc((void**)&d_seed,sizeof(ull));
	cudaMalloc((void**)&d_rad,sizeof(int));
	cudaMalloc((void**)&d_srad,sizeof(int));

	cudaMemcpy(d_sinLUT,&sinLUT,sizeof(int)*1024,cudaMemcpyHostToDevice);
	cudaMemcpy(d_cosLUT,&cosLUT,sizeof(int)*1024,cudaMemcpyHostToDevice);
	cudaMemcpy(d_rad,&rad,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_srad,&srad,sizeof(int),cudaMemcpyHostToDevice);
	//3.5ms; 12us; 3us
	for(ull i = searchFrom; i<searchTo+numBlocks*numThreadsPerBlock;i+=numBlocks*numThreadsPerBlock){
		cudaMemcpy(d_seed,&i,sizeof(ull),cudaMemcpyHostToDevice);
		if(refresh++ % 100000 == 0) cudaDeviceSynchronize();
		checkSeeds<<<numBlocks,numThreadsPerBlock>>>(d_seed, d_sinLUT, d_cosLUT, d_rad, d_srad);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
    	printf("Error: %s\n", cudaGetErrorString(err));
	}

	cudaFree(d_seed);
	cudaFree(d_sinLUT);
	cudaFree(d_cosLUT);
	cudaFree(d_rad);
	cudaFree(d_srad);
	if (printTime) {
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		double seconds = time_span.count();
		printf("%f seconds to calculate (%f seeds per second) with radius %d and small radius %d\n",seconds,(searchTo-searchFrom)/seconds,rad,srad);
	}

	return 0;
}
__global__ void checkSeeds(ull* start, int* d_sinLUT, int* d_cosLUT, int* d_rad, int* d_srad) {
	ull seed, RNGseed1, RNGseed2, chunkseed;
	ll var8, var10;
	int baseX, baseZ, chunkX, chunkZ, angle;
	double dist;
	seed=*start+threadIdx.x+blockIdx.x*numThreadsPerBlock;

	RNGseed2 = seed ^ 25214903917;
	next3(RNGseed2);
	dist = 160+(RNGseed2/2199023255552);
	if(dist > *d_rad*4) return;

	RNGseed1 = seed ^ 25214903917;
	next(RNGseed1);
	var8 = (RNGseed1 >> 16) << 32;
	angle = RNGseed1/274877906944;
	next(RNGseed1);
	var8 += (int) (RNGseed1 >> 16);
	var8 = var8 / 2 * 2 + 1;
	var10 = (RNGseed2 >> 16) << 32;
	next(RNGseed2);
	var10 += (int) (RNGseed2 >> 16);
	var10 = var10 / 2 * 2 + 1;

	baseX = (*(d_cosLUT+angle) * dist) / 8192;
	baseZ = (*(d_sinLUT+angle) * dist) / 8192;


	for (chunkX = baseX - *d_srad; chunkX <= baseX + *d_srad; chunkX++) {
		for (chunkZ = baseZ - *d_srad; chunkZ <= baseZ + *d_srad; chunkZ++) {
			chunkseed = (var8 * chunkX + var10 * chunkZ) ^ (seed ^ 25214903917);
			if (goodChest(chunkseed)) {
				printf("%llu %d %d\n",seed,chunkX,chunkZ);
				goto end;
			}
		}
	}
	end:
		angle = 0;
}

__device__ int nextInt94(ull seed) {
	return (int)(seed >> 17) % 94;
}

__device__ bool goodChest(ull chunkseed) {
	chunkseed *= 270658709695593; //This line and the one after it simulate 130 calls to next()
	chunkseed += 8761842161466;
	chunkseed %= 281474976710656;

	next(chunkseed);
	if(chunkseed < 140737488355328)
		return false;
	next(chunkseed);
	if(nextInt94(chunkseed) > 9)
		return false;
	next(chunkseed);
	if(nextInt94(chunkseed) > 9)
		return false;
	next(chunkseed);
	if(nextInt94(chunkseed) > 9)
		return false;
}
