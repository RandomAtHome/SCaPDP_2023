for PROG in task for
do
for LEVEL in O0 O1 O2 O3
do
	icc -qopenmp -std=c99 -${LEVEL} heat-3d-${PROG}.c heat-3d.h -o heat-3d-${PROG}_${LEVEL}
done
done 
