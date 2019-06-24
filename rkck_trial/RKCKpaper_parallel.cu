// source : https://arxiv.org/pdf/1611.02274.pdf
#include <stdio.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <fstream>
#include <sstream>
using namespace std;

// NEQN - number of ODEs in a system
#define NEQN 3
// NPAR - number of parameters
#define NPAR 3
// THREADS - total number of threads
#define THREADS int(1e2)
// REPSET - number of unique threads repeated
#define REPSET int(1e2)
// constants used in the integrations
#define UROUND (2.22e-16)
#define SAFETY 0.9
#define PGROW (-0.2)
#define PSHRNK (-0.25)
#define ERRCON (1.89e-4)
#define TINY (1.0e-30)
#define P1 0.1

// change parameter file accordingly in parameter_reader function


// function dydt is where the equations are defined. 

// h*derivative -> F
__device__ void dydt(double t, double* y,
 const double* g, double* F, double h){
	F[0] = h * (g[0] * (y[1] - y[0]));
	F[1] = h * (y[0] * (g[1] - y[2]) - y[1]);
	F[2] = h * (y[0] * y[1] - g[2] * y[2]);
}

// functions rkckStep, rkckDriver and intDriver does the integration.
// change function call to appropriate rkcDriver functions in the intDriver function

__device__ void rkckStep(double t, double* y,
	const double* g, double* F, double h,
	double* yTemp, double* yErr){

	// constant values from paper RKCK
	double a[6] = {0.0, 0.2, 0.3, 0.6, 1.0, 0.875};
	double b1[1] = {0.2};
	double b2[2] = {3.0/40, 9.0/40};
	double b3[3] = {3.0/10, -9.0/10, 6.0/5};
	double b4[4] = {-11.0/54, 5.0/2, -70.0/27, 35.0/27};
	double b5[5] = {1631.0/55296, 175.0/512, 575.0/13824, 44275.0/110592, 253.0/4096};
	double c[6] = {37.0/378, 0.0, 250.0/621, 125.0/594, 0.0, 512.0/1771};
	double cs[6] = {2825.0/27648, 0.0, 18575.0/48384, 13525.0/55296, 277.0/14336, 0.25};

	double tmp[NEQN];
	
	double k1[NEQN];
	dydt(t, y, g, k1, h);
	
	double k2[NEQN];
	for(int i = 0; i < NEQN; ++i){
		tmp[i] = b1[0] * k1[i] + y[i];
	}
	dydt(t + a[1] * h, tmp, g, k2, h);
	
	double k3[NEQN];
	for(int i = 0; i < NEQN; ++i){
		tmp[i] = b2[0] * k1[i] + b2[1] * k2[i] + y[i];
	}
	dydt(t + a[2] * h, tmp, g, k3, h);

	double k4[NEQN];
	for(int i = 0; i < NEQN; ++i){
		tmp[i] = b3[0] * k1[i] + b3[1] * k2[i] + b3[2] * k3[i] + y[i];
	}
	dydt(t + a[3] * h, tmp, g, k4, h);

	double k5[NEQN];
	for(int i = 0; i < NEQN; ++i){
		tmp[i] = b4[0] * k1[i] + b4[1] * k2[i] + b4[2] * k3[i] + b4[3] * k4[i] + y[i];
	}
	dydt(t + a[4] * h, tmp, g, k5, h);

	double k6[NEQN];
	for(int i = 0; i < NEQN; ++i){
		tmp[i] = b5[0] * k1[i] + b5[1] * k2[i] + b5[2] * k3[i] + b5[3] * k4[i] + b5[4] * k5[i] + y[i];
	}
	dydt(t + a[5] * h, tmp, g, k6, h);

	for(int i = 0; i < NEQN; ++i){
		yTemp[i] = y[i] + c[0] * k1[i] + c[1] * k2[i] + c[2] * k3[i] + c[3] * k4[i] + c[4] * k5[i] + c[5] * k6[i];
		yErr[i] = yTemp[i] - (y[i] + cs[0] * k1[i] + cs[1] * k2[i] + cs[2] * k3[i] + cs[3] * k4[i] + cs[4] * k5[i] + cs[5] * k6[i]);
	}
}


__device__ void rkckDriver(double t, const double tEnd, 
	double* y, const double* g){
	// max and min step size
	const double hMax = fabs(tEnd - t);
	const double hMin = 1.0e-20;
	const double eps = 1.0e-10;


	// initial step size
	double h = 0.5 * fabs(tEnd - t);

	// integration including
	// trial integration to estimate error
	while(t < tEnd){
		// max step size = remaining time
		h = fmin(tEnd - t, h);

		double yTemp[NEQN], yErr[NEQN];

		// evaluate derivative
		double F[NEQN];
		dydt(t, y, g, F, 1.0);

		rkckStep(t, y, g, F, h, yTemp, yErr);

		// calculate error
		double err = 0.0;
		int nanFlag = 0;
		for(int i = 0; i < NEQN; ++i){
			if(isnan(yErr[i])) nanFlag = 1;

			err = fmax(err, fabs(yErr[i]/(fabs(y[i]) + fabs(h * F[i]) + TINY)));
		}
		err /= eps;

		// if error too large, step decrease, retake
		if ((err > 1.0) || isnan(err) || (nanFlag == 1)) {
			// failed step
			if (isnan(err) || (nanFlag == 1)){

				h *= P1;
			} else {
				h = fmax(SAFETY * h * pow(err, PSHRNK), P1 * h);
			}
		} else{
			// step accepted
			t += h;


			if(err > ERRCON) {
				h = SAFETY * h * pow(err, PGROW); 
			} else{
				h *= 5.0;
			}

			// ensure step size is bounded
			h = fmax(hMin, fmin(hMax, h));

			for(int i = 0; i < NEQN; ++i)
				y[i] = yTemp[i];
		}
	}
}



// general integration kernel
__global__ void
intDriver (const double t, const double tEnd, 
	const int numODE, const double* gGlobal, double* yGlobal){
	// thread ID
	int tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if (tid < numODE){

		// get values from global arrays -> local arrays
		double yLocal[NEQN];
		double gLocal[NPAR];

		for(int i = 0; i < NEQN; ++i){
			yLocal[i] = yGlobal[tid + numODE * i];
		}
		for(int i = 0; i < NPAR; ++i){
			gLocal[i] = gGlobal[tid + numODE * i];
		}

		// call integrator for one timestep
		// replace integrator with rkckDriver or rkcDriver
		// rkck for low stiffness, rkc for high stiffness
		rkckDriver (t, tEnd, yLocal, gLocal);

		// local array -> global array
		for(int i = 0; i < NEQN; ++i){
			yGlobal[tid + numODE * i] = yLocal[i];
		}
	}
}



// to read parameter values from given parameter file
void parameter_reader(double* g, int numODE) { 
  
    // File pointer 
    fstream fin; 
  
    // Open parameter file 
    fin.open("parameters_3_e2.csv", ios::in); 
  
    cout << "Reading parameters" << endl;
    string line, word, temp; 
    int i = 0, j = 0;
    while (fin >> temp) { 
		if(i > numODE){
			cout << "Number of samples read > numODE" << endl;
			break;
		}   

        stringstream s(temp); 
  
        // read every column data of a row and 
        // store it in a string variable, 'word' 
        j = 0;
        while (getline(s, word, ',')) { 
  
            // add all the column data 
            // of a row to a vector 
            if ( j >= NPAR){
            	cout << "Number of parameters read > NPAR" << endl;
            	break;
            }
            g[i + numODE*j] = stof(word);
            j ++;
        }   
        i++;
    }
    fin.close();
    cout << "Done reading parameters" << endl;
} 


int main(){

	// numODE - number of systems
	int numODE = THREADS;

	// // defining y - initial conditions - 2d array with numODE rows and
	// // NEQN columns
	// double *yHost;
	// yHost = (double *) malloc (numODE * NEQN * sizeof(double));

	// for(int i = 0; i < numODE; ++i){
	// 	for(int j = 0; j < NEQN; ++j){
	// 		yHost[i + numODE * j] = y[i][j];
	// 	}
	// }
	double *yHost;
	yHost = (double *) malloc (numODE * NEQN * sizeof(double));

	for(int i = 0; i < numODE; ++i){
		for(int j = 0; j < NEQN; ++j){
			yHost[i + numODE * j] = 0.5;
		}
	}


	// // defining g - parameters - 2d array with numODE rows and
	// // NPAR columns
	// double *gHost;
	// gHost = (double *) malloc (numODE * NPAR * sizeof(double));

	// for(int i = 0; i < numODE; ++i){
	// 	for(int j = 0; j < NPAR; ++j){
	// 		gHost[i + numODE * j] = g[i][j];
	// 	}
	// }
	double *gHost;
	gHost = (double *) malloc (numODE * NPAR * sizeof(double));

	parameter_reader(gHost, numODE);
	// for(int i = 0; i < numODE; ++i){
	// 	gHost[i] = 0.5;
	// 	gHost[i + numODE*1] = 0.5;
	// 	gHost[i + numODE*2] = 1.0 + ((i%REPSET)+1)*1.0/REPSET;
	// }

	int blockSize;
	if(numODE < 4194304) {
		blockSize = 64;
	} else if (numODE < 8388608){
		blockSize = 128;
	} else if (numODE < 16777216){
		blockSize = 256;
	} else {
		blockSize = 512;
	}
	dim3 dimBlock (blockSize, 1);
	dim3 dimGrid (numODE / dimBlock.x + 1, 1);


	// set initial time, end time, initial step size

	double t0 = 0.0;
	double h = 10.0;
	double tEnd = 100.0;
	int numSteps = int((tEnd-t0)/h) + 1;
	double t = t0;
	double tNext = t + h;

	vector< vector< vector<double> > > yvals(numODE, vector< vector<double> >(numSteps, vector<double>(NEQN, 0.0)));


	// start measuring time
	clock_t start_time = clock();
	// setting up GPU global memory arrays
	double *yDevice;
	cudaMalloc ((void**) &yDevice, numODE * NEQN * sizeof(double));

	double *gDevice;
	cudaMalloc ((void**) &gDevice, numODE * NPAR * sizeof(double));

	// CPU to GPU memory transfer
	cudaMemcpy(gDevice, gHost, numODE * NPAR * sizeof(double),
		cudaMemcpyHostToDevice);
	cudaMemcpy (yDevice, yHost, numODE * NEQN * sizeof(double),
		cudaMemcpyHostToDevice);

	int k = 0;
	for(int i = 0; i < numODE; ++i){
		for(int j = 0; j < NEQN; ++j){
			yvals[i][k][j] = yHost[i + numODE*j];
		}
	}
	cout << "GPU kernel calls started" << endl; 

	double progress = 0.0;
	while(t < tEnd){
		// integrating each step till we reach end
		intDriver<<<dimGrid, dimBlock>>> (t, tNext, numODE, gDevice, yDevice);
		cudaMemcpy (yHost, yDevice, numODE * NEQN * sizeof(double),
			cudaMemcpyDeviceToHost);
		++k;
		progress = ((double)k*100.0/(double)numSteps);
		if(k%10 == 0){
			cout << "Progress:\t"<< progress << " %" << endl;
		}
		for(int i = 0; i < numODE; ++i){
			for(int j = 0; j < NEQN; ++j){
				yvals[i][k][j] = yHost[i + numODE*j];
			}
		}

		t = tNext;
		tNext += h;
	}
	clock_t end_time = clock();
	double time_taken = ((double)end_time - (double)start_time)/CLOCKS_PER_SEC;

	cout << "!\n!\n!\nDone!!!" << endl;
	cout << "GPU execution took " << time_taken << " seconds to execute." << endl;
	cudaFree(gDevice);
	cudaFree(yDevice);


	cout << "Printing files to RKCK_parallel_out.csv" << endl;
	cout << numODE << " x " << numSteps << " x " << NEQN << endl;
	clock_t write_start = clock();
	ofstream ofile;
	ofile.open("RKCK_parallel_out.csv");
	for(int i = 0; i < numODE; i++){
		for(int j = 0; j < numSteps; j++){
			for(int m = 0;m < NEQN; m++){
				ofile << yvals[i][j][m];
				if(m != NEQN - 1)
					ofile << ",";
			}
			ofile << endl;
		}
	}
	ofile.close();
	clock_t write_end = clock();
	double write_time = ((double)write_end - (double)write_start)/CLOCKS_PER_SEC;
	cout << "Writing file took " << write_time << " seconds." << endl;

	return 0;
}


























