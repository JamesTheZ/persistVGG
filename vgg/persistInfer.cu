#include "persistInfer.h"

namespace PersistInfer
{
	// TODO: first and last signal value should be in different page to avoid frequent page fault..
	int volatile __managed__ signalIn[20];
	int volatile __managed__ SMs[20];
};
