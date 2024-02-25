for PROG in task for
do
# "((1<<10)+2)" "((1<<11)+2)" "((1<<12)+2)" "((1<<12)+2)"
for LEVEL in 1026 2050 4098 8194
do
	icc -qopenmp -std=c99 -O3 -DN=${LEVEL} var11_${PROG}.c -o var11_${PROG}_${LEVEL}
done
done 
