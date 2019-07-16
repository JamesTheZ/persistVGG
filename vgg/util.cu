#include "util.h"
#include "stdio.h"

struct timeval timeDelta(struct timeval start, struct timeval end)
{
	struct timeval delta;
	delta.tv_sec = end.tv_sec - start.tv_sec;
	delta.tv_usec = end.tv_usec - start.tv_usec;
	if(delta.tv_usec < 0)
	{
		delta.tv_usec += 1e6;
		delta.tv_sec--;
	}

	return delta;
}

void printTime(struct timeval time)
{
	printf("TIME_OF_PIPELINE: %ld.%06ld seconds\n", time.tv_sec, time.tv_usec);
}


