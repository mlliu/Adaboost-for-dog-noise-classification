/*
/*
 * test.c
 *
 *  Created on: Feb 24, 2016
 *      Author: puff
 */


#include<stdio.h>
#include<string.h>
#include<math.h>
#include <malloc.h>
#include"mfcc_adaboost.h"


void main()
 {
	//long long time_start = GetNTime();
	char* wavfile="alaska2.wav";
	char* modelfile="train.txt_1000.model";
	short wavdata[ipframesize];


	model * model_var=GetModel(modelfile);

	TWavHeader waveheader;
	FILE *sourcefile;
	sourcefile=fopen(wavfile,"rb");
	fread(&waveheader,sizeof(struct _TWavHeader),1,sourcefile);
	 //long long time_while = GetNTime();
	while(fread(wavdata,sizeof(short),ipframesize,sourcefile)==ipframesize)
	{
	 //fread(wavdata,sizeof(short),ipframesize,sourcefile);
		GetMfcc(wavdata,ipframesize,model_var);//############################################
		//GetFrame_Result(mfcc_feature,model_var);//############################################


	 //fseek(sourcefile,offset,SEEK_CUR);
	}//离开while loop
	 //long long time_finalresult = GetNTime();
  int label= GetFinal_Result();//############################################

 printf("result is %d\n",label);
 //long long time_end = GetNTime();
 //printf( "all time is %qi\n",time_end-time_start );
 //printf( "finalresult time is %qi\n",time_end-time_finalresult );
 //printf( "time_while time is %qi\n",time_while-time_finalresult );

 }



model * GetModel(char* modelfile)
{
	model* model_var=(model*)malloc(weakclassifier_num*sizeof(model));
	FILE * f;
	int i=0;
	f = fopen(modelfile,"r");
	for(i=0;i<weakclassifier_num;i++)
	{
		fscanf(f,"%f %f %f %f\n",&model_var[i].featureIndex,&model_var[i].threshold,&model_var[i].outputLarger,&model_var[i].outputSmaller);

	}
	fclose(f);
	return model_var;
}
/*
inline unsigned long long GetNTime()
{
       __asm ("RDTSC");
}

*/
