#ifndef PERSIST_INFER_H
#define PERSIST_INFER_H

#define MAX_LAYER_GROUP 20

namespace PersistInfer
{
	extern int volatile __managed__ signalIn[20];
	extern int volatile __managed__ SMs[20];
};

#endif
