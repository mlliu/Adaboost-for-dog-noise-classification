/*
 * test.c
 *
 *  Created on: Feb 23, 2016
 *      Author: puff
 */

#include<stdio.h>
#include<string.h>
#include<math.h>
#include <malloc.h>
#include"mfcc_adaboost.h"


void main()
 {

	char* wavfile="alaska.wav";
	char* modelfile="train.txt_1000.model";
	short wavdata[ipframesize];


	model * model_var=GetModel(modelfile);

	TWavHeader waveheader;
	FILE *sourcefile;
	sourcefile=fopen(wavfile,"rb");
	fread(&waveheader,sizeof(struct _TWavHeader),1,sourcefile);
	while(fread(wavdata,sizeof(short),ipframesize,sourcefile)==ipframesize)
	{

		GetMfcc(wavdata,ipframesize,model_var);
		//GetFrame_Result(mfcc_feature,model_var);//############################################


	 //fseek(sourcefile,offset,SEEK_CUR);
	}//离开while loop

  int label= GetFinal_Result();//############################################
 printf("result is %d",label);
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
