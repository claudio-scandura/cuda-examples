### README Progetto ESP 2010 di Claudio Scandura e Giuseppe Quartarone ### 

ISTRUZIONI PER LA COMPILAZIONE E L'ESECUZIONE

Per la compilazione si consiglia di usare il comando make in quanto si fa uso della compilazione condizionale
per decidere quale dei test deve essere eseguito e su che tipo di dato.
Dare come primo comando “make all” o “make”, in modo da generare gli eseguibili. Successivamente digitare “make test<name>”
con il nome del test da eseguire che deve essere uno tra questi: Scalar, Norm, FIR, FYRSYM, All (per tutti i test).
Se ad esempio si sceglie di eseguire il test sul FIR a coefficienti simmetrici bisogna digitare “make testFIRSYM”.
Infine per ripulire la cartella va dato il comando “make clean”.
Un valore che puo` essere modificato nel makefile e` la variabile TAPS che rappresenta il numero dei coefficienti nel test FIR.
E` anche possibile modificare alcuni valori numerici  (sottoforma di macro) come il numero di thread da eseguire sull'host in caso
di esecuzione parallela (FIR), i semi per la generazione di numeri casuali da inserire nei vettori dei test, il numero di thread 
in un blocco, numero di blocchi e dimensione dati di input. Si consiglia di non modificare tuttavia gli ultimi tre in quanto potrebbero
compromettere il funzionamento dei kernel CUDA. Dopo ogni modifica i file vanno ricompilati con “make all”.
Tutti i test (e i grafici) riportati, sono eseguiti sui valori attuali presenti nei vari file.

Per la compilazione manuale va usato nvcc che ha una sintassi identica a quella di gcc, usare il Makefile come riferimento.


