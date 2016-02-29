
/*
 * mfcc_adaboost.h
 *
 *  Created on: Feb 23, 2016
 *      Author: puff
 */

#ifndef MFCC_ADABOOST_H_
#define MFCC_ADABOOST_H_
#include <stdio.h>
#include <math.h>
//定义bool结 构
typedef enum {false = 0, true = 1} bool;
typedef struct weakclassifiers{
    float featureIndex;
    float threshold;
    float outputLarger;
    float outputSmaller;
	} model;

 #define    ipframesize             256
 #define    fftcoef                 512
 #define    PI                      3.1415926
 #define    TPI                     (2*3.1415926)


 #define    windowsize              30
 #define    rightrate               0.2
 #define    weakclassifier_num      1000
extern float fbank[27];
extern float c[13];
extern float En;
extern float offset;
extern float mfcc_feature[13];
extern int label;

typedef struct _TWavHeader
{
        int rId;
        int rLen;
        int wId;
        int fId;

        int fLen;   //Sizeof(WAVEFORMATEX)

        short wFormatTag;
        short nChannels;
        int nSamplesPerSec;
        int nAvgBytesPerSec;
        short nBlockAlign;
        short wBitsPerSample;
        int dId;
        int wSampleLength;
}TWavHeader;
int GetFinal_Result(void);
void GetFrame_Result(float* mfcc_feature,model* model_var);
void GetMfcc(short *buffer,int len,model *model);
model * GetModel(char* modelfile);
//void GiveVoice(unsigned char *data,int idx,int len);



#endif /* MFCC_ADABOOST_H_ */
