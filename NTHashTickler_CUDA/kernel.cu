// NTHashTickler_CUDA
// By Ryan Ries, 2014, ryanries09@gmail.com, myotherpcisacloud.com
// 

#include <stdio.h>
#include <conio.h>
#include <ctype.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <Windows.h>
#include <helper_cuda_drvapi.h>
#include <drvapi_error_string.h>
#include <device_launch_parameters.h>
#include <bcrypt.h>

int preferredDeviceNum = 0;

bool host_shouldStop = false;
__device__ __managed__ unsigned char randomBytes[32768];
__device__ __managed__ unsigned char inputHashBytes[16] = { 0 };
__device__ __managed__ int maxPasswordLength = 12;
__device__ __managed__ bool device_shouldStop = false;

// These are initialization values for the MD4 hash algorithm. See RFC 1320.
__constant__ uint32_t INIT_A = 0x67452301;
__constant__ uint32_t INIT_B = 0xefcdab89;
__constant__ uint32_t INIT_C = 0x98badcfe;
__constant__ uint32_t INIT_D = 0x10325476;
__constant__ uint32_t SQRT_2 = 0x5a827999;
__constant__ uint32_t SQRT_3 = 0x6ed9eba1;

// Common password characters.
__constant__ char validChars[] = { 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
                                   0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,
                                   0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
                                   0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x4A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F,
                                   0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F,
                                   0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E };

// Does string base begin with substring str?
bool startsWith(char * base, char * str) 
{
	return (strstr(base, str) - base) == 0;
}

// Is this string 32 characters long and only consists of hex chars?
bool IsMD4HashString(char * input)
{
	const char nibbles[] = { "0123456789abcdefABCDEF" };

	if (strlen(input) != 32)
		return false;	

	for (int x = 0; input[x]; x++)
	{
		bool isNibble = false;
		for (int y = 0; y < 16; y++)
		{
			if (nibbles[y] == input[x])
				isNibble = true;
		}
		if (isNibble == false)
			return false;
	}
	return true;
}

// Print some helpful text and exit the program.
void PrintHelpText(char * extraMessage)
{	
	printf("\nNTHashTickler_CUDA v1.0\n");
	printf("-----------------------\n");
	printf("Written by Ryan Ries, myotherpcisacloud.com\n\n");
	printf("Usage: C:\\> NTHashTickler_CUDA hash=d79e1c308aa5bbcdeea8ed63df412da9\n");
	printf("                               [maxlen=8]\n");
	printf("                               [preferredDevice=0]\n");
	printf("\nUses brute force to find a plain text input that generates an\n");
	printf("NT hash that matches the one supplied by the user. An NT (or NTLM)\n");
	printf("hash is the MD4 hash of the Unicode little endian plain text.\n");
	printf("Arguments are not case sensitive.\n\n");
	printf("The maxlen argument is optional and specifies the maximum password\n");
	printf("length for which to generate hashes. Default is 12.\n\n");
	printf("The preferredDevice argument specifies the NVIDIA CUDA device to use.\n");
	printf("Only a single CUDA device is supported at this time because I only had\n");
	printf("one to test with. Default is 0 (the first CUDA device found) but you\n");
	printf("can override that if your second or third device is faster.\n");
	printf("\n%s\n", extraMessage);
	exit(EXIT_FAILURE);
}

__device__ void NTHash(char * password, int length, uint32_t * output)
{
	uint32_t nt_buffer[16] = { 0 };
	int i = 0;
	
	//The length of key need to be <= 27
	for (; i<length / 2; i++)
		nt_buffer[i] = password[2 * i] | (password[2 * i + 1] << 16);

	// Padding
	if (length % 2 == 1)
		nt_buffer[i] = password[length - 1] | 0x800000;
	else
		nt_buffer[i] = 0x80;

	nt_buffer[14] = length << 4;

	uint32_t a = INIT_A;
	uint32_t b = INIT_B;
	uint32_t c = INIT_C;
	uint32_t d = INIT_D;

	/* Round 1 */
	a += (d ^ (b & (c ^ d))) + nt_buffer[0]; a = (a << 3) | (a >> 29);
	d += (c ^ (a & (b ^ c))) + nt_buffer[1]; d = (d << 7) | (d >> 25);
	c += (b ^ (d & (a ^ b))) + nt_buffer[2]; c = (c << 11) | (c >> 21);
	b += (a ^ (c & (d ^ a))) + nt_buffer[3]; b = (b << 19) | (b >> 13);

	a += (d ^ (b & (c ^ d))) + nt_buffer[4]; a = (a << 3) | (a >> 29);
	d += (c ^ (a & (b ^ c))) + nt_buffer[5]; d = (d << 7) | (d >> 25);
	c += (b ^ (d & (a ^ b))) + nt_buffer[6]; c = (c << 11) | (c >> 21);
	b += (a ^ (c & (d ^ a))) + nt_buffer[7]; b = (b << 19) | (b >> 13);

	a += (d ^ (b & (c ^ d))) + nt_buffer[8]; a = (a << 3) | (a >> 29);
	d += (c ^ (a & (b ^ c))) + nt_buffer[9]; d = (d << 7) | (d >> 25);
	c += (b ^ (d & (a ^ b))) + nt_buffer[10]; c = (c << 11) | (c >> 21);
	b += (a ^ (c & (d ^ a))) + nt_buffer[11]; b = (b << 19) | (b >> 13);

	a += (d ^ (b & (c ^ d))) + nt_buffer[12]; a = (a << 3) | (a >> 29);
	d += (c ^ (a & (b ^ c))) + nt_buffer[13]; d = (d << 7) | (d >> 25);
	c += (b ^ (d & (a ^ b))) + nt_buffer[14]; c = (c << 11) | (c >> 21);
	b += (a ^ (c & (d ^ a))) + nt_buffer[15]; b = (b << 19) | (b >> 13);

	/* Round 2 */
	a += ((b & (c | d)) | (c & d)) + nt_buffer[0] + SQRT_2; a = (a << 3) | (a >> 29);
	d += ((a & (b | c)) | (b & c)) + nt_buffer[4] + SQRT_2; d = (d << 5) | (d >> 27);
	c += ((d & (a | b)) | (a & b)) + nt_buffer[8] + SQRT_2; c = (c << 9) | (c >> 23);
	b += ((c & (d | a)) | (d & a)) + nt_buffer[12] + SQRT_2; b = (b << 13) | (b >> 19);

	a += ((b & (c | d)) | (c & d)) + nt_buffer[1] + SQRT_2; a = (a << 3) | (a >> 29);
	d += ((a & (b | c)) | (b & c)) + nt_buffer[5] + SQRT_2; d = (d << 5) | (d >> 27);
	c += ((d & (a | b)) | (a & b)) + nt_buffer[9] + SQRT_2; c = (c << 9) | (c >> 23);
	b += ((c & (d | a)) | (d & a)) + nt_buffer[13] + SQRT_2; b = (b << 13) | (b >> 19);

	a += ((b & (c | d)) | (c & d)) + nt_buffer[2] + SQRT_2; a = (a << 3) | (a >> 29);
	d += ((a & (b | c)) | (b & c)) + nt_buffer[6] + SQRT_2; d = (d << 5) | (d >> 27);
	c += ((d & (a | b)) | (a & b)) + nt_buffer[10] + SQRT_2; c = (c << 9) | (c >> 23);
	b += ((c & (d | a)) | (d & a)) + nt_buffer[14] + SQRT_2; b = (b << 13) | (b >> 19);

	a += ((b & (c | d)) | (c & d)) + nt_buffer[3] + SQRT_2; a = (a << 3) | (a >> 29);
	d += ((a & (b | c)) | (b & c)) + nt_buffer[7] + SQRT_2; d = (d << 5) | (d >> 27);
	c += ((d & (a | b)) | (a & b)) + nt_buffer[11] + SQRT_2; c = (c << 9) | (c >> 23);
	b += ((c & (d | a)) | (d & a)) + nt_buffer[15] + SQRT_2; b = (b << 13) | (b >> 19);

	/* Round 3 */
	a += (d ^ c ^ b) + nt_buffer[0] + SQRT_3; a = (a << 3) | (a >> 29);
	d += (c ^ b ^ a) + nt_buffer[8] + SQRT_3; d = (d << 9) | (d >> 23);
	c += (b ^ a ^ d) + nt_buffer[4] + SQRT_3; c = (c << 11) | (c >> 21);
	b += (a ^ d ^ c) + nt_buffer[12] + SQRT_3; b = (b << 15) | (b >> 17);

	a += (d ^ c ^ b) + nt_buffer[2] + SQRT_3; a = (a << 3) | (a >> 29);
	d += (c ^ b ^ a) + nt_buffer[10] + SQRT_3; d = (d << 9) | (d >> 23);
	c += (b ^ a ^ d) + nt_buffer[6] + SQRT_3; c = (c << 11) | (c >> 21);
	b += (a ^ d ^ c) + nt_buffer[14] + SQRT_3; b = (b << 15) | (b >> 17);

	a += (d ^ c ^ b) + nt_buffer[1] + SQRT_3; a = (a << 3) | (a >> 29);
	d += (c ^ b ^ a) + nt_buffer[9] + SQRT_3; d = (d << 9) | (d >> 23);
	c += (b ^ a ^ d) + nt_buffer[5] + SQRT_3; c = (c << 11) | (c >> 21);
	b += (a ^ d ^ c) + nt_buffer[13] + SQRT_3; b = (b << 15) | (b >> 17);

	a += (d ^ c ^ b) + nt_buffer[3] + SQRT_3; a = (a << 3) | (a >> 29);
	d += (c ^ b ^ a) + nt_buffer[11] + SQRT_3; d = (d << 9) | (d >> 23);
	c += (b ^ a ^ d) + nt_buffer[7] + SQRT_3; c = (c << 11) | (c >> 21);
	b += (a ^ d ^ c) + nt_buffer[15] + SQRT_3; b = (b << 15) | (b >> 17);

	output[0] = a + INIT_A;
	output[1] = b + INIT_B;
	output[2] = c + INIT_C;
	output[3] = d + INIT_D;
}

__global__ void KernelMain()
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	int randPasswordLength = tid % maxPasswordLength + 1;	
	char * randomString = new char[randPasswordLength + 1];
	uint32_t buffer[4] = { 0 };
	unsigned char hashOut[16] = { 0 };	

	randomString[randPasswordLength] = 0;
	for (int x = 0; x < randPasswordLength; x++)	
		randomString[x] = validChars[randomBytes[tid + x] % sizeof(validChars)];	

	// Buffer comes out in 4 8-byte parts, reversed.
	NTHash(randomString, randPasswordLength, buffer);
	delete randomString;

	hashOut[3]  = (buffer[0] & 0xFF000000UL) >> 24;
	hashOut[2]  = (buffer[0] & 0x00FF0000UL) >> 16;
	hashOut[1]  = (buffer[0] & 0x0000FF00UL) >> 8;
	hashOut[0]  = (buffer[0] & 0x000000FFUL);

	hashOut[7]  = (buffer[1] & 0xFF000000UL) >> 24;
	hashOut[6]  = (buffer[1] & 0x00FF0000UL) >> 16;
	hashOut[5]  = (buffer[1] & 0x0000FF00UL) >> 8;
	hashOut[4]  = (buffer[1] & 0x000000FFUL);

	hashOut[11] = (buffer[2] & 0xFF000000UL) >> 24;
	hashOut[10] = (buffer[2] & 0x00FF0000UL) >> 16;
	hashOut[9]  = (buffer[2] & 0x0000FF00UL) >> 8;
	hashOut[8]  = (buffer[2] & 0x000000FFUL);

	hashOut[15] = (buffer[3] & 0xFF000000UL) >> 24;
	hashOut[14] = (buffer[3] & 0x00FF0000UL) >> 16;
	hashOut[13] = (buffer[3] & 0x0000FF00UL) >> 8;
	hashOut[12] = (buffer[3] & 0x000000FFUL);

	bool match = true;
	for (int x = 0; x < 16; x++)
	{
		if (hashOut[x] != inputHashBytes[x])
		{
			match = false;
			break;
		}
	}

	if (!device_shouldStop)
	if (match)
	{
		device_shouldStop = true;
		printf("\nMatch Found on Thread %d!\n\n%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x = %s\n\n", tid, hashOut[0], hashOut[1],   hashOut[2],  hashOut[3], 
			                                                                                                                    hashOut[4], hashOut[5],   hashOut[6],  hashOut[7],
																								                                hashOut[8], hashOut[9],   hashOut[10], hashOut[11],
																								                                hashOut[12], hashOut[13], hashOut[14], hashOut[15], randomString);
		
	}	
}

// This is the "press any key to quit" interrupt thread.
DWORD WINAPI Interrupt(LPVOID lpParam)
{
	_getch();	
	host_shouldStop = true;	
	return 0;
}

int main(int argc, char **argv)
{
	unsigned char dummyArray[16] = { 0 };
	CUdevice device;
	int deviceCount = 0;
	char deviceName[256];
	CUresult error_id;
	size_t totalGlobalMem;
	int multiProcessorCount = 0;
	int majorCapabilityVersion = 0, minorCapabilityVersion = 0;
	int clockRate = 0, memoryClock = 0, memBusWidth = 0;
	int maxThreadsPerMultiProcessor = 0, maxThreadsPerBlock = 0;
	int maxBlockDim[3], maxGridDim[3];	
	unsigned __int64 startingTime = 0, endingTime = 0, hostTimerfrequency = 0;
	unsigned __int64 hashesGenerated = 0;
	

	printf("\n");

	// Wrong number of command line args, exit.
	if (argc < 2 || argc > 4)
		PrintHelpText("Must supply at least 1 command line argument and no more than 3!");	

	// Loop through all args, convert to lowercase, and validate for correctness.
	for (int x = 1; x < argc; x++)
	{
		int equalsSignsInArg = 0;
		for (int c = 0; c < strlen(argv[x]); c++)
		{
			argv[x][c] = tolower(argv[x][c]);
			if (argv[x][c] == '=')
				equalsSignsInArg++;

		}
		if (equalsSignsInArg != 1)		
			PrintHelpText("Argument syntax error - number of equals signs was not 1!");		

		// hash= is the only mandatory arg that the user must supply.
		if (startsWith(argv[x], "hash="))
		{
			// Everything is offset by 5 because we are skipping the first 5 bytes (hash=).
			if (strlen(argv[x]) != 37)
				PrintHelpText("The supplied hash has incorrect syntax!(Invalid length?)");
			
			// Room for 32 characters plus a null terminator.
			char hashStr[33] = { '\0' };
			for (int v = 5; v < 37; v++)			
				hashStr[v - 5] = argv[x][v];			

			if (!IsMD4HashString(hashStr))			
				PrintHelpText("The supplied hash has incorrect syntax! (Invalid chars?)");			

			// Now convert the hex string into a byte array by grabing two characters at a time and converting them to base 16 numbers.
			for (int j = 0; j < 32; j += 2)
			{
				char b[2] = { 0, 0 };
				b[0] = (char)hashStr[j];
				b[1] = (char)hashStr[j + 1];
				inputHashBytes[j / 2] = (char)strtoul(b, NULL, 16);
			}
			printf("  Searching for Hash: %s\n", hashStr);
		}
		// This command line argument is optional. Everything is offset by 7 characters because we are skipping the first 7 bytes (maxlen=).
		else if (startsWith(argv[x], "maxlen="))
		{
			char maxLenArgBuf[8];
			for (int v = 7; v < 15; v++)			
				maxLenArgBuf[v - 7] = argv[x][v];
			
			maxPasswordLength = strtoul(maxLenArgBuf, NULL, 10);
			if (maxPasswordLength < 1 || maxPasswordLength > 120)			
				PrintHelpText("Max Password Length (maxlen) was out of range!");

		}
		// This command line argument is optional.
		else if (startsWith(argv[x], "preferreddevice="))
		{
			char preferredDeviceBuf[8];
			for (int v = 16; v < 20; v++)			
				preferredDeviceBuf[v - 16] = argv[x][v];
			
			preferredDeviceNum = strtoul(preferredDeviceBuf, NULL, 10);
			if (preferredDeviceNum < 0 || preferredDeviceNum > 1024)			
				PrintHelpText("Preferred Device was out of range!");

		}		
		else
		{
			PrintHelpText("Unrecognized command line argument!");
		}
	}	

	printf(" Max Password Length: %i\n", maxPasswordLength);

	// User apparently did not supply a hash, cannot continue.
	if (memcmp(inputHashBytes, dummyArray, 16) == 0)	
		PrintHelpText("Hash is missing!");

	printf("\n");

	// Must initialize the CUDA driver API first.
	error_id = cuInit(0);
	if (error_id != CUDA_SUCCESS)
	{
		printf("\nERROR: NVIDIA CUDA initialization returned %s!\n", getCudaDrvErrorString(error_id));
		return 1;
	}
	error_id = cuDeviceGetCount(&deviceCount);
	if (error_id != CUDA_SUCCESS)
	{
		printf("\nERROR: cuDeviceGetCount reutnred %s!\n", getCudaDrvErrorString(error_id));
		return 1;
	}

	printf("  CUDA Devices Found: %d\n", deviceCount);
	if (deviceCount < 1)
		return 1;
	
	if (preferredDeviceNum > deviceCount - 1)
	{
		printf("\nERROR: The specified preferred device number (%d) was not found!\n", preferredDeviceNum);
		return 1;
	}

	device = preferredDeviceNum;

	// Collect more statistics about the chosen CUDA device.
	error_id = cuDeviceGetName(deviceName, 256, device);
	if (error_id != CUDA_SUCCESS)
	{
		printf("\nERROR: cuDeviceGetName returned %s!\n", getCudaDrvErrorString(error_id));
		return 1;
	}

	error_id = cuDeviceComputeCapability(&majorCapabilityVersion, &minorCapabilityVersion, device);
	if (error_id != CUDA_SUCCESS)
	{
		printf("ERROR: cuDeviceComputeCapability returned %s!\n", getCudaDrvErrorString(error_id));
		return 1;
	}
	
	error_id = cuDeviceTotalMem(&totalGlobalMem, device);
	if (error_id != CUDA_SUCCESS)
	{
		printf("\nERROR: cuDeviceTotalMem returned %s!\n", getCudaDrvErrorString(error_id));
		return 1;
	}
	getCudaAttribute<int>(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
	getCudaAttribute<int>(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
	getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
	getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);
	getCudaAttribute<int>(&maxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
	getCudaAttribute<int>(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
	getCudaAttribute<int>(&maxBlockDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);
	getCudaAttribute<int>(&maxBlockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device);
	getCudaAttribute<int>(&maxBlockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device);
	getCudaAttribute<int>(&maxGridDim[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
	getCudaAttribute<int>(&maxGridDim[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device);
	getCudaAttribute<int>(&maxGridDim[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device);	

	// Display the rest of the specs on the chosen CUDA device.
	printf("Selected Device Name: %s\n", deviceName);
	printf("  Capability Version: %d.%d\n", majorCapabilityVersion, minorCapabilityVersion);
	printf("       Device Memory: %.0f MB\n", (float)totalGlobalMem / 1048576.0f);
	printf("     Multiprocessors: %d\n", multiProcessorCount);
	printf("       CUDA Cores/MP: %d\n", _ConvertSMVer2CoresDRV(majorCapabilityVersion, minorCapabilityVersion));
	printf("    Total CUDA Cores: %d\n", _ConvertSMVer2CoresDRV(majorCapabilityVersion, minorCapabilityVersion) * multiProcessorCount);
	printf("      GPU Clock Rate: %.0f MHz\n", clockRate * 1e-3f);
	printf("   Memory Clock Rate: %.0f MHz\n", memoryClock * 1e-3f);
	printf("    Memory Bus Width: %d bit\n", memBusWidth);
	printf("      Max Threads/MP: %d\n", maxThreadsPerMultiProcessor);
	printf("   Max Threads/Block: %d\n", maxThreadsPerBlock);
	printf("Max Thread Block Dim: x=%d,y=%d,z=%d\n", maxBlockDim[0], maxBlockDim[1], maxBlockDim[2]);
	printf(" Max Grid Dimensions: x=%d,y=%d,z=%d\n", maxGridDim[0],   maxGridDim[1], maxGridDim[2]);

	// Setting up our high precision timer.
	QueryPerformanceFrequency((LARGE_INTEGER *)&hostTimerfrequency);
	if (hostTimerfrequency < 1)
	{
		printf("\nERROR: Unable to query performance frequency!\n");
		return 1;
	}
	printf("\nHashing will now commence. Press any key to interrupt the program.\n");
	QueryPerformanceCounter((LARGE_INTEGER *)&startingTime);
	// Start the "Press any key to interrupt... thread"
	CreateThread(NULL, 0, Interrupt, NULL, 0, NULL);
	
	dim3 gridDimensions(64, 64, 1);	// x * y * z = the number of blocks being launched, but z must be 1 for compute capability 1.x devices.
	dim3 blockDimensions(8, 8, 8);	// x * y * z = the number of threads per block. Probably should not exceed maxThreadsPerBlock here.

	while (true)
	{	
		if (host_shouldStop || device_shouldStop)
			break;

		// Must generate some GOOD random bytes.		
		if (BCryptGenRandom(NULL, (PBYTE)randomBytes, sizeof(randomBytes), BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0)
		{
			printf("\nERROR: BCryptGenRandom Error!\n");
			break;
		}
		
		KernelMain <<< gridDimensions, blockDimensions >>>();
		if (cudaDeviceSynchronize() != CUDA_SUCCESS)
		{
			printf("\nERROR: CUDA Kernel Error!\n");
			break;			
		}
		
		hashesGenerated += ((gridDimensions.x * gridDimensions.y * gridDimensions.z) * (blockDimensions.x * blockDimensions.y * blockDimensions.z));
		//printf("%lu\n", hashesGenerated);
	}				
	
	QueryPerformanceCounter((LARGE_INTEGER *)&endingTime);

	cudaDeviceReset();
	
	double totalTimeElapsed = (double)(endingTime - startingTime) / hostTimerfrequency;
	printf("\n%lu hashes generated in %.2f seconds.\n", hashesGenerated, totalTimeElapsed);
	printf("%.0f hashes/second.\n", hashesGenerated / totalTimeElapsed);
    
	return 0;
}