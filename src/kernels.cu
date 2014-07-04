/*
 *	File: kernels.cu
 *	Contiene i kernel da eseguire sul device e alcune macro
 *	Autori: Claudio Scandura, Giuseppe Quartarone
 */

/* Macro per la gestione degli errori */
#define safe_call(x) \
	if( (x) != cudaSuccess ) { fprintf(stderr,"ERRORE CUDA: %s\n", cudaGetErrorString(cudaGetLastError())); return -1; }
#define ec_null(x) \
	if ( (x) == NULL ) { perror("ERRORE"); return -1; }
#define ec_rv(N)\
	if ((int)N) { errno=N; perror("Errore"); exit(errno); }

#define TABLE_BAR "\t\t\t\t|++++++++++|++++++++++++++|++++++++++++++++|++++++++++++|+++++++++++++++++|\n"

#define THREAD_FIR 512			/* Numero di thread in un blocco per FIR (no change) */
#define THREAD_SCALAR 256		/* Numero di thread in un blocco per prodotto scalare (no change)*/


#ifndef TAPS
/*	
	Calcola il prodotto scalare di n coppie di vettori <V,W>.
	Ogni blocco di thread elabora un prodotto scalare;
	i thread in un blocco lavorano su fette di elementi della coppia <V,W>

	a:	puntatore a memoria globale del device; 
		contiene V0,...,Vn allocati in modo contiguo
	
	b:	come a, ma contiene i vettori W0,...,W1

	res:	zona di memoria globale dove scrivere i risultati

	vectors: il numero dei vettori

	elems:	lunghezza dei vettori	
*/
__global__ void ScalarProduct(float *a, float* b, float* res, int vectors, int elems){
	/* memoria shared per scrivere i risultati dei singoli thread */
	__shared__ float th_r[THREAD_SCALAR];
	/* memoria shared per scrivere i risultati parziali delle singole fette*/
	__shared__ float sl_r[THREAD_SCALAR];
	int slices=elems/THREAD_SCALAR;		/* le fette sono piu` di una quando gli elementi sono piu dei thread nel blocco
											NB: slices non deve superare il numero dei thread in un blocco */
	const int tId = threadIdx.x;
	/* scorre i vettori in a e b */
	for(int vec_id = blockIdx.x; vec_id < vectors; vec_id += gridDim.x){
		int base = elems * vec_id;
		/* scandisce le fette dei due vettori */
		for(int slice = 0; slice < slices; slice++, base += blockDim.x){
			th_r[tId] = a[base + tId] * b[base + tId];
			__syncthreads();
			/* calcola il risultato finale della fetta */
			for( int step = blockDim.x /2; step > 0; step /= 2){
				if( tId < step )
					th_r[tId] += th_r[step + tId];
				__syncthreads();
			}
			/* il primo thread del blocco scrive il risultato parziale della fetta */
			if ( tId == 0) 
				sl_r[slice] = th_r[0];
		}
		/* somma i risultati delle fette */
		for(int step = slices/2; step > 0; step /=2){
			if( tId < step )
				sl_r[tId] += sl_r[step + tId];
			__syncthreads();
		}
		/* il risultato finale viene scritto in memoria globale */
		if( tId == 0 )
			res[vec_id] = sl_r[0];
	}
}

/*
	Funzione analoga a quella sopra, calcola la norma 2 di n vettori,
	non occorre quindi avere i vettori W
*/
__global__ void norma2(float *a, float* res, int vectors, int elems){
	__shared__ float th_r[THREAD_SCALAR];
	__shared__ float sl_r[THREAD_SCALAR];
	int slices=elems/THREAD_SCALAR;
	const int tId = threadIdx.x; /* identificatore thread nel blocco */
	for(int vec_id = blockIdx.x; vec_id < vectors; vec_id += gridDim.x){
		int base = elems * vec_id;
		for(int slice = 0; slice < slices; slice++, base += blockDim.x){
			th_r[tId] = a[base + tId] * a[base + tId];
			__syncthreads();
			for( int step = blockDim.x /2; step > 0; step /= 2){
				if( tId < step )
					th_r[tId] += th_r[step + tId];
				__syncthreads();
			}
			if ( tId == 0) 
				sl_r[slice] = th_r[0];
		}
		for(int step = slices/2; step > 0; step /=2){
			if( tId < step )
				sl_r[tId] += sl_r[step + tId];
			__syncthreads();
		}
		if( tId == 0 )
			res[vec_id] = sqrtf(sl_r[0]);
	}
}

#else
	#ifdef __SYM__										/* Tramite le direttive di compilazione si alloca meno memoria in caso di FIR simmetrico */
		__device__ __constant__ float d_coeffs[TAPS/2];
	#else
		__device__ __constant__ float d_coeffs[TAPS];
	#endif

/*	Texture reference alla delay line per velocizzare
	l' accesso in memoria globale  */
texture<float, 1, cudaReadModeElementType> texRef1;

/*
	Calcolo FIR
	A blocchi di thread corrispondono blocchi della delay line,
	ogni thread quindi calcola il risultato di una singola posizione
	del vettore dei risultati. 
	Si usa la tecnica del buffer circolare per implementare la convoluzione
	res: la delay line
	size: la dimensione di res
*/
__global__ void fir(float* res, int size){
	const int tId = threadIdx.x;	/* identificatore del thread nel blocco */
	register float t_res;		/* registro per l'accumulo del risultato del thread */
	/* scorre i blocchi della delay line */
	for(int b_id = blockIdx.x; b_id < size/THREAD_FIR ; b_id += gridDim.x){
		t_res=0.0;
		/* ricava la posizione di res sul quale calcolare la convoluzione */
		int element = THREAD_FIR * b_id +tId, count=element;
		for (int k=0; k<TAPS; k++) {
			/* preleva la posizione di res dalla memoria texture (cached) */
			t_res+=d_coeffs[k]*tex1Dfetch(texRef1, count--);
			if (count<0)
				count=size-1;
		}
		res[element]=t_res;
	}
}

/*
	Come fir() con la differenza che si assume di avere un
	vettore dei coefficienti simmetrico
*/
__global__ void fir_sym(float* res, int size){
	const int tId = threadIdx.x;
	register float t_res;
	for(int b_id = blockIdx.x; b_id < size/THREAD_FIR ; b_id += gridDim.x){
		t_res=0.0;
		int element = THREAD_FIR * b_id +tId, count=element, sym_ind;
		/*	calcola l'indirizzo simmetrico di res[count]
			relativamente alla lunghezza di input response */
		if (count-TAPS+1<0)
			sym_ind=size+(count - TAPS+1);
		else
			sym_ind=count-TAPS+1;
		for (int k=0; k<TAPS/2; k++) {
			/*	essendo count e sym_ind generalmente non contigui
				questi due prelievi sono la causa della lentezza di questo kernel
				rispetto a fir() poiche` generano texture miss */
			t_res+=d_coeffs[k]*(tex1Dfetch(texRef1, count--)+tex1Dfetch(texRef1, sym_ind++));
			if (count<0) count=size-1;
			if (sym_ind>=size) sym_ind=0;
		}
		res[element]=t_res;
	}
}
#endif

/*
	Funzione per l'inizializzazione casuale di vettori
*/
static void init(float *v, int size, float seed) {
	int k;
	srand(time(NULL));
	for (k=0; k<size; k++)
		v[k]=random()*seed;
}

