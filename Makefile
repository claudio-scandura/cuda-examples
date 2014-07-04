###############################################################
###############################################################
## Makefile progetto ESP
##
###############################################################
###############################################################


# Compiler
CC = nvcc

.PHONY: all clean testScalar testNorm testFIR testFIRSYM testAll

# Numero coefficienti per FIR, non deve superare 8192
TAPS=8192

# Compila tutti i sorgenti
# NB: make all deve essere eseguito prima dei test
all: 
	$(CC) scalarTest.cu -o scalarTest
	$(CC) scalarTest.cu -o normTest -D __NORM__
	$(CC) FIR_Test.cu -o FIRTest -D TAPS=$(TAPS)
	$(CC) FIR_Test.cu -o FIR_SymTest -D TAPS=$(TAPS) -D __SYM__
	@echo -e 'MAKE ALL completato\a'

clean:
	-rm -fr scalarTest normTest FIRTest FIR_SymTest *~  

testScalar: 
	./scalarTest

testNorm: 
	./normTest

testFIR: 
	./FIRTest

testFIRSYM: 
	./FIR_SymTest

testAll:
	./scalarTest
	./normTest
	./FIRTest
	./FIR_SymTest

