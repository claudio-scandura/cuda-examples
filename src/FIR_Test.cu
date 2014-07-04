/*
 *	File: FIR_Test.cu
 *	Contiene le esecuzioni per i vari test del filtro FIR e FIR simmetrico
 *	Autori: Claudio Scandura, Giuseppe Quartarone
 */
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>
#include <errno.h>

#include <kernels.cu>

#define FIR "  FIR  "
#define FIR_SYM "Sym FIR"


#define SEED1 2.000540006			/* Semi per la generazione di numeri da inserire nei vettori */
#define SEED2 4.00370028

#define TESTS 10			/* Numero di test da effettuare */

#define P_THREAD	8	/* Numero di thread da eseguire sulla CPU */

static int testId;			/* Specifica quale test effettuare (=0) fir, (!=0) fir simmetrico */

const int d_line_size[TESTS]				/* dimensione delay line durante il test i-esimo */
		 ={8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304}; 

const int blocks[TESTS]
		={ 16, 16, 32, 32, 64, 64, 128, 128, 256, 256};	/* Numero di blocchi cuda da eseguire nel kernel durante l'i-esimo test,
														   NB: deve essere una potenza di due minore o uguale di d_line_size[i]/THREAD_FIR per ogni i,
														   (almeno pari al numero dei MP del device per una buona utilizzazione dell'hardware) */
#ifdef __SYM__
	float	im_response[TAPS/2];			/* Il vettore dei coefficienti */
#else 
	float im_response[TAPS];
#endif

float	*d_line = NULL,				/* il vettore contenente la delay line */
		
		*result = NULL;				/* il risultato dell'operazione di filtraggio */

/*
	Convoluzione su una o piu posizioni della delay line.
	I thread che eseguono questa funzione ricevono in ingresso una partizione,
	e l'intera lunghezza, della delay line sulla quale calcolare la convoluzione.
	il vettore d_line viene gestito come buffer circolare
*/
void *conv(void* args);

/*
	Uguale alla funzione conv() ma assume che input response sia simmetrico
	e quindi lavora sulla prima meta` dei coefficienti
*/
void *conv_sym(void* args);

/**
	Esegue il test del filtro FIR e FIR simmetrico a seconda del valore di
	testId. Al variare di attempt variano i dati
*/
static int test(int attempt);

/*
	Effettua un test del FIR o FIR simmetrico, a seconda delle direttive di compilazione
	Il test consiste in piu` tentativi che differiscono per la quantita` di dati da
	usare come imput. 
	Oltre i valori impostati come limite non e` garantita il completamento dell'esecuzione
*/
int main(int argc, char *argv[]){
	#ifdef __SYM__			/* Decide quale dei due test eseguire */
		 testId=1;
	#else
		testId=0;
	#endif
	int i=0;
	for( i=0; i<TESTS; i++){
		printf("\t\t\t###########################>>> %s Test (%d) <<<###############################\n"
				"\t\t\tMemory used: %d Bytes  Delay line length: %d  Input response length: %d\n\t\t\tCPU Threads: %d  Blocks: %d  Threads per block: %d\n\n",
				(testId==0 ? FIR : FIR_SYM), i+1, (d_line_size[i]+(testId==0 ? TAPS : TAPS/2))*sizeof(float),d_line_size[i], TAPS, P_THREAD, blocks[i],THREAD_FIR);
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
		free(d_line);
		free(result);
		printf("Done!\n");
	}
	return 0;
}

static int test(int attempt){
	void *status;
	pthread_t *workers;						/* I thread per l'esecuzione parallela dell'algoritmo */
	float *d_Dline, *d_res, *c_results=NULL				/* Variabili da allocare sul device  per il trasferimento dati */
			
			,time_transfer_to, time_transfer_from, time_alloc,
			time_ker;											/* Variabili per monitorare i ritardi di allocazione, inizializzazione, */
	int elapsed_alloc, elapsed_init, elapsed_ex;				/* trasferimento dati (nel caso di CUDA) ed esecuzione FIR */
	int i, k, slice, *tmp, *args,
	coeff_size=TAPS;						
	struct timeval start, end;
	/*  HOST 
		Allocazione memoria sull'host e inizializzazione vettori
	 */
	if (testId!=0)						/* lavora sulla prima meta` del vettore coefficienti nel caso FIR simmetrico */
		coeff_size/=2;
	gettimeofday(&start, NULL);
	ec_null( (d_line=(float*)realloc(d_line, d_line_size[attempt]*sizeof(float)))) 
	ec_null( (result=(float*)realloc(result, d_line_size[attempt]*sizeof(float))))
	ec_null((workers=(pthread_t *)malloc(sizeof(pthread_t)*P_THREAD)))
	gettimeofday(&end, NULL);
	elapsed_alloc=1000000*(end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec);
	gettimeofday(&start, NULL);	
	if (attempt==0)
		init(im_response,coeff_size, SEED1);
	init(d_line,d_line_size[attempt], SEED2); 
	gettimeofday(&end, NULL);	
	elapsed_init=1000000*(end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec);
	/* 
		Esecuzione FIR sull' host (CPU)
	*/	
	slice=d_line_size[attempt]/P_THREAD;
	gettimeofday(&start, NULL);	
	ec_null(  (tmp=args=(int*)malloc(sizeof(int)*P_THREAD*3)))
	/* Crea i thread e li assegna a diversi blocchi della delay line */
	for (i=0,k=0; i<d_line_size[attempt]; i+=slice, k++) {
		args[0]=i;
		args[1]=i+slice-1;
		args[2]=d_line_size[attempt];
		if (testId==0){
			ec_rv(pthread_create(&(workers[k]), NULL, conv, args))
		}
		else {
			ec_rv(pthread_create(&(workers[k]), NULL, conv_sym, args))
		}
		args+=3;
	}
	/* Aspetta la terminazione dei thread workers */
	for (k=0; k<P_THREAD; k++){
		ec_rv(pthread_join(workers[k], &status));
	}
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
	ec_null( (c_results=(float*)malloc(d_line_size[attempt]*sizeof(float))))
	safe_call( cudaEventCreate(&c_start))
	safe_call( cudaEventCreate(&c_end) )
	safe_call( cudaEventRecord(c_start,0))
	safe_call( cudaMalloc((void **)&d_Dline, d_line_size[attempt]*sizeof(float)))
	safe_call( cudaMalloc((void **)&d_res, d_line_size[attempt]*sizeof(float)))
	safe_call( cudaEventRecord(c_end,0))
	safe_call( cudaEventSynchronize(c_end))
	safe_call( cudaEventElapsedTime(&time_alloc,c_start,c_end))
	/*
		Trasferimento delay line e input response da host a device
	*/
	safe_call( cudaEventCreate(&c_start))
	safe_call( cudaEventCreate(&c_end) )
	safe_call( cudaEventRecord(c_start,0))
	if (attempt==0)
		safe_call( cudaMemcpyToSymbol(d_coeffs, im_response, sizeof(float)*coeff_size))
	safe_call( cudaMemcpy(d_Dline,d_line,d_line_size[attempt]*sizeof(float),cudaMemcpyHostToDevice) )
	safe_call( cudaBindTexture(0, texRef1, d_Dline, d_line_size[attempt]*sizeof(float)))
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
		fir<<<blocks[attempt], THREAD_FIR>>>(d_res, d_line_size[attempt]);
	else
		fir_sym<<<blocks[attempt], THREAD_FIR>>>(d_res, d_line_size[attempt]);
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
	safe_call( cudaMemcpy(c_results, d_res, d_line_size[attempt]*sizeof(float),cudaMemcpyDeviceToHost))
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
	for(int j=0; j<d_line_size[attempt]; j++) { 
		relative+=fabs((c_results[j]-result[j]) / result[j]);}
	printf("\t\t\tEpsilon Average: %.20f ", relative/d_line_size[attempt]);
	printf("\n\t\t\tOK\n\n");
	/*
		Cleanup
	*/
	free(workers);
	free(tmp);
	free(c_results);
	safe_call( cudaFree(d_Dline))
	safe_call( cudaFree(d_res))
	return 0;
}

void *conv(void* args) {
	int *from_to=(int*)args;
	float res;
	int i, k, count, from=from_to[0], to=from_to[1], size=from_to[2];
	/* scandisce la partizione */
	for (i=from; i<=to; i++) {
		res=0.0;
		count = i;
		/* scorre il vettore dei coefficienti e accumula i risultati */
		for (k=0; k<TAPS; k++) {
			res+=im_response[k]*d_line[count--];
			if (count<0) count =size-1;
		}
		result[i]=res;
	}
	return (void *)0;
}

void *conv_sym(void* args) {
	int *from_to=(int*)args;
	int i, k, count, sym_ind, from=from_to[0], to=from_to[1], size=from_to[2];
	float res;
	for (i=from; i<=to; i++) {
		res=0.0;
		count = i;
		/*	calcola l'indirizzo simmetrico di d_line[count]
			relativamente alla lunghezza di im_response */
		if (count-TAPS+1<0)
			sym_ind=size+(count - TAPS+1);
		else
			sym_ind=count-TAPS+1;
		/* Scorre la prima meta` dei coefficienti */
		for (k=0; k<TAPS/2; k++) {

			res+=(im_response[k]*(d_line[count--]+d_line[sym_ind++]));
			if (count<0) count =size-1;
			if (sym_ind>=size) sym_ind=0;
		}
		result[i]=res;
	}
	return (void *)0;
}














