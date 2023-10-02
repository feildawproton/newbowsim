#include <omp.h>
#include <stdlib.h>
#include <assert.h>

#include <stdio.h> //please remove when we are done debugging


/*
 * N_orig is the length of the original time array (Ts)
 * N_new is the length of the interpolation (T_interpolated)
 * Ts_k_indices keeps track of which indices in Ts are to the left of entries in T_interpolated
 * function writes to T_interpolated and Ts_k_indices
 * time needs to be sorted for time scale calc to work
 * combine #pragma omp parallel for and #pragma omp for into #pragma omp parallel for/
 */
void 
interpolate_T_omp(const unsigned N_orig     , const unsigned N_new   , 
	          const float    *T         , 
		        float    *T_intrpltd,       unsigned *T_k_ndc)
{
	assert(N_new >= N_orig);
	float t_scale = T[N_orig-1] - T[0];

	unsigned nthreads = omp_get_num_threads();
	printf("we will be using %i threads\n", nthreads); 
	printf("PLEASE REMOVE THESE DEBUG STATEMENTS");
	#pragma omp parallel
	{	
		unsigned nthreads = omp_get_num_threads();
		printf("we are using %i threads\n", nthreads);
		unsigned tid = omp_get_thread_num();
		printf("hello from thread %i\n", tid);
		printf("PLEASE REMOVE THE DEBUG STATEMENTS!!!");
		#pragma omp for
		for(unsigned n = 0; n < N_new; n++)
		{
			// -- INTERPOLATE TIME -- //
			// we are adding the original times back in
			// so why did Thomas or Jason undo then redo this?
			float nf    = (float)n;
			float Nf    = (float)N_new;
			float t_val = ((nf / (Nf - 1.0)) * t_scale) + T[0];	
			
			// -- LEFT ENTRY IN ORIGINAL DATA FOR EACH NEW POINT -- //
			unsigned k           = 0;
			//inner loop not parallel.  besides, we are writing to a single int
			for(unsigned i = 0; i < N_orig; i++)
			{
				if(T[i] < t_val)
					k = i;
			}
		
			T_k_ndc[n]    = k;
			T_intrpltd[n] = t_val;
		}
	}
}

/*
 * calculates the slopes at each point of the original length vectors
 * with a restriction to keep the result monotonic
 * should we do parallel loops for this one?
 */
void 
calc_M(const unsigned N, const float *X, const float *Y, float *M)
{
	float *S = (float*)malloc((N-1)*sizeof(float));

	// -- SLOPES -- //
	for(unsigned i = 0; i < N-1; i++)
	{
		float dy = Y[i+1] - Y[i];
		float dx = X[i+1] - X[i];
		if(dx == 0.0)
			S[i] = 0.0;
		else
			S[i] = dy / dx;
	}	

	// -- DEGREE ONE COEFFICIENTS -- //
	M[0]   = S[0];
	M[N-1] = S[N-1];
	for(unsigned i = 0; i < N - 2; i++)
	{
		// hmmmmmmm
		if(S[i] * S[i-1] <= 0.0)
			M[i] = 0.0;
		else
			M[i] = (S[i - 1] + S[i]) / 2.0;
	}

	// -- 2ND and 3RD ORDER COEFFICIENTS -- //
	for(unsigned i = 0; i < N; i++)
	{
		float alpha_i = 0.0;
		float beta_i  = 0.0;

		if(S[i] != 0.0)
		{
			alpha_i = M[i]     / S[i];
			beta_i  = M[i + 1] / S[i];
		}

		if     (alpha_i < 0.0)
			M[i]    = 0.0;
		else if(alpha_i > 3.0)
			M[i]    = 3.0 * S[i];

		if     (beta_i < 0.0)
			M[i+1] = 0.0;
		else if(beta_i > 3.0)
			M[i+1] = 3.0 * S[i];
	}	
	
	free(S);
}

// -- STATIC FUNCS -- //
/*
 * static in c means keep to this file
 * from Cubic Hermite Spline article on wikipedia
 */
static float _h_00(const float t)
{
       	float  h_t =  2.0*t*t*t - 3.0*t*t + 1.0;
	return h_t;
}
static float _h_10(const float t)
{
	float  h_t =  1.0*t*t*t - 2.0*t*t + t;
	return h_t;
}
static float _h_01(const float t)
{
	float  h_t = -2.0*t*t*t + 3.0*t*t;
	return h_t;
}
static float _h_11(const float t)
{
	float  h_t =  1.0*t*t*t - 1.0*t*t;
	return h_t;
}
void interpolate_Y_omp(const unsigned N_orig     , const unsigned N_new      ,
   	               const float    *T         , const float    *Y         , const float *M, 
		       const unsigned *T_k_ndc   , const float    *T_intrpltd,       
		       float          *Y_intrpltd)
{
	#pragma omp parallel for
	for(unsigned n = 0; n < N_new; n++)
	{
		unsigned k = T_k_ndc[n];
		assert(k < N_orig);
		float T_k  = T[k];
		float T_k1 = T[k + 1];
		float dt   = T_k1 - T_k;
		float T_n  = T_intrpltd[n];
		float t    = (T_n - T_k) / dt;
		float Y_k  = Y[k];
		float M_k  = M[k];
		float Y_k1 = Y[k + 1];
		float M_k1 = M[k + 1];
		float h_00 = _h_00(t);
		float h_10 = _h_10(t);
		float h_01 = _h_01(t);
		float h_11 = _h_11(t);

		float Y_n  = (Y_k*h_00) + (dt*M_k*h_10) + (Y_k1*h_01) + (dt*M_k1*h_11);

		Y_intrpltd[n] = Y_n;
	}
}














