/*
 * mfcc_adaboost.c
 *
 *  Created on: Feb 23, 2016
 *      Author: puff
 */

#include<stdio.h>
#include<string.h>
#include<math.h>
//#include <malloc.h>
#include <stdlib.h>
#include"mfcc_adaboost.h"


 float fbank[27];
 float c[13];
 float En;
 float offset=-256;
 float mfcc_feature[13];

 int label=0;

 int frame_index=0;
 int frame_result[500];
 static void GenCepWin (float *cw,int cepLiftering, int count);
 //计算并返回score,输入mfcc_feature[39],和model_var[weakclassifier_var]
 static double evaluate(float *mfcc_feature,model* model_var);
 //predict,输入mfcc_feature和model_file,返回1--dog,0--noise
 static int predict(float* mfcc_feature,model* model_var);
 static void PreEmphasise (float *s, float k);
 static void Ham (float *s,int frameSize);
 static void initfft(float* data,float* fftbuf);
 static void Realft (float *s);
 static void FFT(float *s, int invert);
 static float Mel(int k,float fres);
 static void Wave2FBank(float *fftbuf);
 static void FBank2MFCC(int n);
 static void CptEn(float *s);
 static void WeightCepstrum (int start, int count, int cepLiftering);


void GetMfcc(short *buffer,int len,model *model)
{
    if (len !=ipframesize)
        printf("the length of wav data should be 256");
    else
    {
        float* data=(float*)malloc((ipframesize+1)*sizeof(float));
        float* fftbuf=(float*)malloc((fftcoef)*sizeof(float));
        int i=0;
        for(i=0;i<ipframesize;i++) data[i+1]=buffer[i];
        float PREEMCOEF=0.97;
        PreEmphasise (data, PREEMCOEF);
        Ham (data, ipframesize);
        CptEn(data);
        initfft(data,fftbuf);
        free(data);
        Realft (fftbuf);
        Wave2FBank(fftbuf);
        FBank2MFCC(12);
        WeightCepstrum (1, 12, 22);

         //predict the result for every frame
         int j=1;
         mfcc_feature[0]=En;
         for (j=1;j<=12;j++)
         {
             mfcc_feature[j]=c[j];
         }
    }

    GetFrame_Result(mfcc_feature,model);
}

void GetFrame_Result(float* mfcc_feature,model* model_var)
{
	 frame_index++;//count num
	 frame_result[frame_index]=predict(mfcc_feature,model_var);
		//printf("mfcc_feature[1] is %f\n",mfcc_feature[1]);
		// printf("frame_result[frame_index] is %d %d\n",frame_index,frame_result[frame_index]);
}
int GetFinal_Result()
{
	 label=0;
	 int index_i=0;
	 for(index_i=0;index_i<frame_index-windowsize;index_i++)
	 {
		 int sum=0,index_j=0;
		 float sum_thresold=rightrate*windowsize;
		 for(index_j=index_i;index_j<index_i+windowsize;index_j++) sum=sum+frame_result[index_j];
		 //printf("sum is %d\n",sum);printf("sum_thresold is %f\n",sum_thresold);
		 if (sum >sum_thresold) {label=1;break;}
	 }
	 return label;
}

void PreEmphasise (float *s, float k)
{
   int i;
   float preE;

   preE = k;
   for (i=ipframesize;i>=2;i--)
      s[i] -= s[i-1]*preE;
   s[1] *= 1.0-preE;
}


/* EXPORT->Ham: Apply Hamming Window to Speech frame s */
void Ham (float *s,int frameSize)
{
	int i;
    float a;
    float* hamWin=(float*)malloc((ipframesize+1)*sizeof(float));
    a = TPI / (frameSize - 1);
    for (i=1;i<=frameSize;i++)
       hamWin[i] = 0.54 - 0.46 * cos(a*(i-1));

    for (i=1;i<=frameSize;i++)
       s[i] *= hamWin[i];
    free(hamWin);
}




/* EXPORT-> FFT: apply fft/invfft to complex s */
void FFT(float *s, int invert)
{
   int ii,jj,n,nn,limit,m,j,inc,i;
   double wx,wr,wpr,wpi,wi,theta;
   double xre,xri,x;

   n=fftcoef;
   nn=n / 2; j = 1;
   for (ii=1;ii<=nn;ii++) {
      i = 2 * ii - 1;
      if (j>i) {
         xre = s[j]; xri = s[j + 1];
         s[j] = s[i];  s[j + 1] = s[i + 1];
         s[i] = xre; s[i + 1] = xri;
      }
      m = n / 2;
      while (m >= 2  && j > m) {
         j -= m; m /= 2;
      }
      j += m;
   };
   limit = 2;
   while (limit < n) {
      inc = 2 * limit; theta = TPI / limit;
      if (invert) theta = -theta;
      x = sin(0.5 * theta);
      wpr = -2.0 * x * x; wpi = sin(theta);
      wr = 1.0; wi = 0.0;
      for (ii=1; ii<=limit/2; ii++) {
         m = 2 * ii - 1;
         for (jj = 0; jj<=(n - m) / inc;jj++) {
            i = m + jj * inc;
            j = i + limit;
            xre = wr * s[j] - wi * s[j + 1];
            xri = wr * s[j + 1] + wi * s[j];
            s[j] = s[i] - xre; s[j + 1] = s[i + 1] - xri;
            s[i] = s[i] + xre; s[i + 1] = s[i + 1] + xri;
         }
         wx = wr;
         wr = wr * wpr - wi * wpi + wr;
         wi = wi * wpr + wx * wpi + wi;
      }
      limit = inc;
   }
   if (invert)
      for (i = 1;i<=n;i++)
         s[i] = s[i] / nn;

}

/* EXPORT-> Realft: apply fft to real s */
void Realft (float *s)
{
   int n, n2, i, i1, i2, i3, i4;
   float xr1, xi1, xr2, xi2, wrs, wis;
   float yr, yi, yr2, yi2, yr0, theta, x;

   n=fftcoef/ 2; n2 = n/2;
   theta = PI / n;
   FFT(s,0);
   x = sin(0.5 * theta);
   yr2 = -2.0 * x * x;
   yi2 = sin(theta); yr = 1.0 + yr2; yi = yi2;
   for (i=2; i<=n2; i++) {
      i1 = i + i - 1;      i2 = i1 + 1;
      i3 = n + n + 3 - i2; i4 = i3 + 1;
      wrs = yr; wis = yi;
      xr1 = (s[i1] + s[i3])/2.0; xi1 = (s[i2] - s[i4])/2.0;
      xr2 = (s[i2] + s[i4])/2.0; xi2 = (s[i3] - s[i1])/2.0;
      s[i1] = xr1 + wrs * xr2 - wis * xi2;
      s[i2] = xi1 + wrs * xi2 + wis * xr2;
      s[i3] = xr1 - wrs * xr2 + wis * xi2;
      s[i4] = -xi1 + wrs * xi2 + wis * xr2;
      yr0 = yr;
      yr = yr * yr2 - yi  * yi2 + yr;
      yi = yi * yr2 + yr0 * yi2 + yi;
   }
   xr1 = s[1];
   s[1] = xr1 + s[2];
   s[2] = 0.0;
}





void initfft(float* data,float* fftbuf)
{
  int k;
  for (k=1; k<=ipframesize; k++)
	 fftbuf[k] = data[k];    /* copy to workspace */
   for (k=ipframesize+1; k<=fftcoef; k++)
      fftbuf[k] = 0.0;   /* pad with zeroes */

}

void Wave2FBank(float *fftbuf)
{
	float loWt[ipframesize+1];
	float loChan[ipframesize+1];
	float melk;
	float ms=3177.54;
	float cf[28];
	float fres=0.0307967;
	int k,chan;
	int klo=2;
	int khi=ipframesize;
    int maxChan=27;

   for (chan=1; chan <= maxChan; chan++)
   {
    cf[chan] = ((float)chan/(float)maxChan)*ms ;
   }
   for (k=1,chan=1; k<=ipframesize; k++)
   {
      melk = Mel(k,fres);
      if (k<klo || k>khi)
		  loChan[k]=-1;
      else
	  {
         while (cf[chan] < melk  && chan<=maxChan) ++chan;
         loChan[k] = chan-1;
      }
   }
   for (k=1; k<=ipframesize; k++)
   {
      chan = loChan[k];
      if (k<klo || k>khi)
		 loWt[k]=0.0;
      else
	  {
         if (chan>0)
            loWt[k] = ((cf[chan+1] - Mel(k,fres)) /
                          (cf[chan+1] - cf[chan]));
         else
            loWt[k] = (cf[1]-Mel(k,fres))/(cf[1]);
      }

   }
   //wave2bank
   const float melfloor = 1.0;
   int bin;
   float t1,t2;   /* real and imag parts */
   float ek;      /* energy of k'th fft channel */

   for (k = 2; k <= ipframesize-1; k++)
   {             /* fill bins */
        t1 = fftbuf[2*k-1]; t2 = fftbuf[2*k];

        ek = sqrt(t1*t1 + t2*t2);
        bin = loChan[k];
        t1 = loWt[k]*ek;
        if (bin>0) fbank[bin] += t1;
        if (bin<26) fbank[bin+1] += ek - t1;
    }

      /* Take logs */
     // if (info.takeLogs)
   for (bin=1; bin<=26; bin++)
   {
        t1 = fbank[bin];
        if (t1<melfloor) t1 = melfloor;
        fbank[bin] = log(t1);
   }
}



float Mel(int k,float fres)
{
   return 1127 * log(1 + (k-1)*fres);
}


/* EXPORT->FBank2MFCC: compute first n cepstral coeff */
void FBank2MFCC(int n)
{
   int j,k,numChan;
   float mfnorm,pi_factor,x;

   numChan =26;
   mfnorm = sqrt(2.0/(float)numChan);
   pi_factor = PI/(float)numChan;
   for (j=1; j<=n; j++)  {
      c[j] = 0.0; x = (float)j * pi_factor;
      for (k=1; k<=numChan; k++)
         c[j] += fbank[k] * cos(x*(k-0.5));
      c[j] *= mfnorm;
   }
}







/* ¼ÆËãÃ¿Ö¡µÄÄÜÁ¿ */
void CptEn(float *s)
{

     En = 0.0;
     int k=1;
    for (k=1; k<=ipframesize; k++)
      En += (s[k]*s[k]);
	En=log(En);

}


void WeightCepstrum (int start, int count, int cepLiftering)
{
   int i,j;
   float cepWin[13];
   //if (cepWinL != cepLiftering || count > cepWinSize)
   GenCepWin(cepWin, cepLiftering,count);
   j = start;
   for (i=1;i<=count;i++)
      c[j++] *= cepWin[i];
}




/* GenCepWin: generate a new cep liftering vector */
static void GenCepWin (float *cw,int cepLiftering, int count)
{
   int i;
   float a, Lby2;

  // if (cepWin==NULL || VectorSize(cepWin) < count)
   //   cepWin = CreateVector(&sigpHeap,count);
   a = PI/cepLiftering;
   Lby2 = cepLiftering/2.0;
   for (i=1;i<=count;i++)
      cw[i] = 1.0 + Lby2*sin(i * a);

}

int predict(float* mfcc_feature,model* model_var)
{
	float score = evaluate(mfcc_feature,model_var);
	int frame_result=0;
	if (score>0)
		frame_result=1;
	else frame_result=0;
	return frame_result;
}

double evaluate(float *mfcc_feature,model* model_var)
{
	int i=0;
	float score=0;
    for (i = 0; i < weakclassifier_num; ++i)//waek_classfier_model_index
    {
    	int feature_index=model_var[i].featureIndex;

    	if ( mfcc_feature[feature_index] > model_var[i].threshold )
    		score += model_var[i].outputLarger;
    	else score += model_var[i].outputSmaller;
    }
    return score;
}




