for PROG in task for
do
for LEVEL in O0 O1 O2 O3
do
	icc -qopenmp -std=c99 -${LEVEL} var11_${PROG}.c -o var11_${PROG}_${LEVEL}
done
done 
