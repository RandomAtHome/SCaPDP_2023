for PROG in task for
do
for LEVEL in MINI_DATASET SMALL_DATASET MEDIUM_DATASET LARGE_DATASET EXTRALARGE_DATASET
do
	icc -qopenmp -std=c99 -O3 -D${LEVEL} heat-3d-${PROG}.c heat-3d.h -o heat-3d-${PROG}_${LEVEL}
done
done 
