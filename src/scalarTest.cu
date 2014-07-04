/*
 *	File: scalarTest.cu
 *	Contiene le esecuzioni per i vari test sull'operazione di prodotto scalare e norma 2 di vettori
 *	Autori: Claudio Scandura, Giuseppe Quartarone
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <errno.h>

#include <kernels.cu>

#define SCALAR "Scalar"
#define NORM "Norm 2"

#define SEED1 0.6			/* Semi per la generazione di numeri da inserire nei vettori */
#define SEED2 0.28
#define TESTS 12			/* Numero di test da effettuare */
#define SCALAR_LIMIT 11			/*	Limite ai test effettuabili sul kernel per il prodotto scalare */

static int testId;			/* Specifica quale test effettuare (=0) prodotto scalare, (!=0) norma2 */

const int num_elem[TESTS]				/* Quantita` di elementi in un vettore durante il test i-esimo */
		 ={ 512, 1024, 2048, 2048, 4096, 8192, 8192, 16384, 16384, 32768, 65536,/*norm*/ 65536};
const int blocks[TESTS]
		={ 4 , 8, 16, 32, 32, 64, 128, 128, 256, 512,512,/*norm*/ 512};			/* Numero di blocchi cuda da eseguire nel kernel durante l'i-esimo test,
																				   NB: deve essere una potenza di due minore o uguale di vectors[i] per ogni i,
																				   (almeno pari al numero dei MP del device per una buona utilizzazione dell'hardware) */
const int vectors[TESTS]
		={ 4, 8, 16, 64, 128, 128, 256, 256, 512, 1024, 1024,/*norm*/2048};			/* Numero di (coppie di) vettori durante l'i-esimo test */


float	*v_vectors = NULL,				/* Insieme dei vettori, V={v0,..,vn-1} n=vectors[k] per ogni k,
										   allocati in blocchi contigui di dimensione num_elem[k] per ogni k*/
		 
		*w_vectors = NULL,				/* Come sopra ma per i vettori di W */
		
		*results = NULL;				/* Vettore risultati results=[( V0 x W0^T ),...,( Vn-1 x Wn-1^T )] n=vectors[k] per ogni k (prodotto scalare)
											results=[||V0||,...,||Vn||] (norma 2)
										*/

/*
	Calcola un prodotto scalare su una partizione dei vettori a e b,
	base e lim indicano l'inizio e la fine della partizione
	Se il parametro b e` NULL allora viene calcolata la norma 2 della partizione
*/
static float scalarProduct(float* a, float *b, int base, int lim);

/**
	Esegue il test del prodotto scalare o norma a seconda del valore di
	testId. Al variare di attempt variano i dati
*/
static int test(int attempt);

/*
	Effettua un test del prodotto scalare o norma, a seconda delle direttive di compilazione
	Il test consiste in piu` tentativi che differiscono per la quantita` di dati da
	usare come imput. 
	Oltre i valori impostati come limite non e` garantita il completamento dell'esecuzione
*/
int main(int argc, char *argv[]){
	#ifdef __NORM__			/* Decide quale dei due test da eseguire */
		 testId=1;
	#else
		testId=0;
	#endif
	int i=0, bound=(testId==0?SCALAR_LIMIT:TESTS);
	int  long data_size;
	for( i=0; i<bound; i++){
		data_size=vectors[i]*num_elem[i];
		printf("\t\t\t###########################>> %s Test (%d) <<<###############################\n"
				"\t\t\tMemory used: %d Bytes\tVectors: %d\tVectors size: %d\n\t\t\tBlocks: %d\tThreads per block: %d\tSlices: %d\n\n",
				(testId==0 ? SCALAR : NORM), i+1, ((data_size*2)+vectors[i])*sizeof(float),vectors[i], num_elem[i], blocks[i],THREAD_SCALAR,num_elem[i]/THREAD_SCALAR);
		if( test(i) == -1 ) {
			fprintf(stderr,"\t\t\tTest %d failed!\n",i+1);
			return -1;
		}
	}
	/*
		Cleanup variabili globali
	*/
	if (i>0){
		printf("\n\t\t\t\t\t\t\t\t\t\t\tTests passed!\nCleaning up...\n");
		free(v_vectors);
		if (testId==0)
			free(w_vectors);
		free(results);
		printf("Done!\n");
	}
	return 0;
}

static int test(int attempt){
	
	float *d_a, *d_res, *d_b, *c_results=NULL				/* Variabili da allocare sul device  per il trasferimento dati */
			
			,time_transfer_to, time_transfer_from, time_alloc,
			time_ker;											/* Variabili per monitorare i ritardi di allocazione, inizializzazione, */
	int elapsed_alloc, elapsed_init, elapsed_ex;				/* trasferimento dati (nel caso di CUDA) ed esecuzione prodotto scalare */
	
	int long data_size=vectors[attempt]*num_elem[attempt];	/* Numero totale di elementi in v_vectors/w_vectors */
	int i, k;
	struct timeval start, end;
	/*  HOST 
		Allocazione memoria sull'host e inizializzazione vettori
	 */
	gettimeofday(&start, NULL);
	ec_null( (v_vectors = (float *)realloc(v_vectors, data_size*sizeof(float))))
	if (testId==0) 
		ec_null( (w_vectors=(float*)realloc(w_vectors, data_size*sizeof(float)))) 
	ec_null( (results=(float*)realloc(results, vectors[attempt]*sizeof(float))))
	gettimeofday(&end, NULL);
	elapsed_alloc=1000000*(end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec);
	gettimeofday(&start, NULL);	
	init(v_vectors,data_size, SEED1);
	if (testId==0) 
		init(w_vectors,data_size, SEED2); 
	gettimeofday(&end, NULL);	
	elapsed_init=1000000*(end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec);
	/* 
		Esecuzione prodotto scalare sull' host (CPU)
	*/	
	gettimeofday(&start, NULL);	
	for(i=0, k=0; i<data_size; i+=num_elem[attempt],k++)			/* Se testId!=0 ==> W_vectors==NULL, esegue norma2 */
		results[k]=scalarProduct(v_vectors,w_vectors,i,i+num_elem[attempt]);
	gettimeofday(&end, NULL);
	elapsed_ex=1000000*(end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec);
	/*
		Stampa report
	*/
	printf("%s\t\t\t\t|          |  ALLOCATION  |      INIT      |  TRANSFER  |    EXECUTION    |\n%s", TABLE_BAR, TABLE_BAR);
	printf("\t\t\t\t|   CPU    |%8.4fms    |%10.4fms    |     N/A    |%11.4fms    |\n", (float)elapsed_alloc/1000, (float)elapsed_init/1000,
			(float)elapsed_ex/1000);
	printf("%s", TABLE_BAR);
	/*  CUDA 
		Allocazione memoria sul device
	*/
	cudaEvent_t c_start,c_end;
	ec_null( (c_results=(float*)realloc(c_results, vectors[attempt]*sizeof(float))))
	safe_call( cudaEventCreate(&c_start))
	safe_call( cudaEventCreate(&c_end) )
	safe_call( cudaEventRecord(c_start,0))
	safe_call( cudaMalloc((void **)&d_a, data_size*sizeof(float)))
	if (testId==0) 
		safe_call( cudaMalloc((void **)&d_b, data_size*sizeof(float))) 
	safe_call( cudaMalloc((void **)&d_res, vectors[attempt]*sizeof(float)))
	safe_call( cudaEventRecord(c_end,0))
	safe_call( cudaEventSynchronize(c_end))
	safe_call( cudaEventElapsedTime(&time_alloc,c_start,c_end))
	/*
		Trasferimento vettori da host a device
	*/
	safe_call( cudaEventCreate(&c_start))
	safe_call( cudaEventCreate(&c_end) )
	safe_call( cudaEventRecord(c_start,0))
	safe_call( cudaMemcpy(d_a,v_vectors,data_size*sizeof(float),cudaMemcpyHostToDevice))
	if (testId==0) 
		safe_call( cudaMemcpy(d_b,w_vectors,data_size*sizeof(float),cudaMemcpyHostToDevice)) 
	safe_call( cudaEventRecord(c_end,0))
	safe_call( cudaEventSynchronize(c_end))
	safe_call( cudaEventElapsedTime(&time_transfer_to,c_start,c_end))
	/*
		Esecuzione del kernel
	*/
	safe_call( cudaEventCreate(&c_start))
	safe_call( cudaEventCreate(&c_end) )
	safe_call( cudaEventRecord(c_start,0))
	if (testId==0)
		ScalarProduct<<<blocks[attempt], THREAD_SCALAR>>>(d_a,d_b,d_res,vectors[attempt], num_elem[attempt]);
	else
		norma2<<<blocks[attempt], THREAD_SCALAR>>>(d_a, d_res, vectors[attempt], num_elem[attempt]);
	safe_call( cudaGetLastError());
	safe_call( cudaEventRecord(c_end,0))
	safe_call( cudaEventSynchronize(c_end))
	safe_call( cudaEventElapsedTime(&time_ker,c_start,c_end))
	/*
		Copia dei risultati da device a host
	*/
	safe_call( cudaEventCreate(&c_start))
	safe_call( cudaEventCreate(&c_end) )
	safe_call( cudaEventRecord(c_start,0))
	safe_call( cudaMemcpy(c_results, d_res, vectors[attempt]*sizeof(float),cudaMemcpyDeviceToHost))
	safe_call( cudaEventRecord(c_end,0))
	safe_call( cudaEventSynchronize(c_end))
	safe_call( cudaEventElapsedTime(&time_transfer_from,c_start,c_end))
	/*
		Stampa report
	*/
	printf("\t\t\t\t|   GPU    |%8.4fms    |%10.4fms    |%9.4fms |%11.4fms    |\n",time_alloc,
		(float)elapsed_init/1000, time_transfer_from+time_transfer_to, time_ker);
	printf("%s", TABLE_BAR);
	/*
		Calcolo errore relativo medio sui risultati calcolati dal device
		rispetto a quelli dell'host
	*/
	float relative=0.0;
	printf("\n\t\t\tValidating results...\n");
	for(int j=0; j<vectors[attempt]; j++) 
		relative+=fabs((c_results[j]-results[j]) / results[j]);
	printf("\t\t\tEpsilon Average: %.20f ", relative/vectors[attempt]);
	printf("\n\t\t\tOK\n\n");
	/*
		Cleanup
	*/
	free(c_results);
	safe_call( cudaFree(d_a))
	if (testId==0) 
		safe_call( cudaFree(d_b)) 
	safe_call( cudaFree(d_res))
	return 0;
}

static float scalarProduct(float* a, float *b, int base, int lim){
	int i;
	float res=0.0;
	float *vect=b;
	if (b==NULL) vect=a;
	for(i=base; i<lim; i++){
		res+= a[i]*vect[i];
	}
	if (b==NULL)
		return sqrtf(res);
	return res;
}















