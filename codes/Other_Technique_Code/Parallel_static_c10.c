#include<stdio.h>
#include"omp.h"
#include<stdlib.h>
#include<time.h>
float alpha;
int num_iter;
int m=1000,n=500;
float** X;
float* Y;
float* theta;

//*************************find error for given data after calculating theta*******************************************
float finderror(){
	int iter,i,j,k;
	double sum;
	float error=0;
	float A[m];
	
	for(i=0;i<m;i++){
		sum=0.0;
		for(k=0;k<n;k++)
			sum += ((X[i][k])*theta[k]);
			A[i] = sum-Y[i];
			
	}
	
		
	for(i=0;i<m;i++)
		error += A[i]*A[i];
	
	error = error/m;
	return error;

}

//********************check for invertability********************************************
int dothis(float *X,float *Y,int n,int i){
	int p,q;
	int state=0;
	float tmp[n];
	for(p=i+1;p<n;p++){
		if(*(X+p*n+p)!=0){
			for(q=0;q<n;q++)
				tmp[q] = *(X+i*n+q);
			for(q=0;q<n;q++)
				*(X+i*n+q) = *(X+p*n+q);
			for(q=0;q<n;q++)
				*(X+p*n+q) = tmp[q];
			state=1;
			break;	
		}
	}
	return state;
}
//***********************************inverse the matrix*********************************

int inv(float *A,float *B,int n){
	int i,j,k,w;
	float I[n][n];
	float X[n][n];
	float scalex,scalei,s;
	#pragma omp parallel for schedule(static,10) shared(n,X,A) private(i,j) 
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			X[i][j] = *(A+i*n+j);
		}
	}
	
//******************************Identity matrix******************************************
	#pragma omp parallel for schedule(static,10) shared(n,I) private(i,j) 	
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			if(i==j)
				I[i][j] = 1;
			else
				I[i][j] = 0;
		}
	}

//****************************Gaussian elimination process to find inverse of Matrix X*************************************	
	for(i=0;i<n;i++){
		s = X[i][i];
		if(s==0.0){
			if(!dothis(X,I,n,i)){
				return 0;
			}	
		}
		#pragma omp parallel for schedule(static,10) shared(n,X,I,i,s) private(j) 
		for(j=0;j<n;j++){
			X[i][j]/=s;
			I[i][j]=I[i][j]/s;
		}

		#pragma omp parallel for schedule(static,10) shared(n,X,I,i) private(k,j,scalex,scalei) 
		for(j=i+1;j<n;j++){
			scalex = -X[j][i]/X[i][i];
			scalei = scalex;
			
			for(k=0;k<n;k++){
				X[j][k] += X[i][k]*scalex;
				I[j][k] += I[i][k]*scalei;
			}		
		}
					

	}

	for(i=n-1;i>0;i--){
		#pragma omp parallel for schedule(static,10) shared(n,X,I,i) private(s,j,k) 
		for(j=i-1;j>=0;j--){
			s = -X[j][i]/X[i][i];
						
			for(k=0;k<n;k++){
				X[j][k] += X[i][k]*s;
				I[j][k] += I[i][k]*s;
			}
		}
	} 

	#pragma omp parallel for schedule(static,10) shared(n,B,I) private(i,j) 
	for(i=0;i<n;i++){
		for(j=0;j<n;j++)
			*(B+i*n+j) = I[i][j];
	}

}



void gradientDescent(){
	int iter,i,j,k;
	float sum;
	float A[n][n];
	float c[n];
	float Xt[n][m];
	float B[n][n];
	float D[n][m];
	omp_set_num_threads(4);
//****************transpose of X ---> Xt ************************************
	#pragma omp parallel for schedule(static,10) shared(m,n,X,Xt) private(i,j) 
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){
			Xt[j][i] = X[i][j];
		}
	}
	
//*************************   Xt*X   ****************************************** 	
	#pragma omp parallel for schedule(static,10) shared(m,n,X,Xt,A) private(i,j,k,sum) 	
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			sum=0.0;
			for(k=0;k<m;k++)
				sum += Xt[i][k]*(X[k][j]);	
			A[i][j] = sum;
			
		}
	}

//************************* Inverse of matrix  *********************************

	
	if(inv(A,B,n)==0){
		
		return;
	}
int p,q;
	/*for(p=0;p<n;p++){
		for(q=0;q<n;q++)
			printf("%f ",B[p][q]);
		printf("\n");
	}*/

//***************************  B*Xt  *****************************************
	#pragma omp parallel for schedule(static,10) shared(m,n,B,Xt,D) private(i,j,k,sum) 	
	for(i=0;i<n;i++){
		for(j=0;j<m;j++){
			sum=0.0;
			for(k=0;k<n;k++)
				sum += B[i][k]*Xt[k][j];	
			D[i][j] = sum;
		}
	}
	
	/*for(p=0;p<n;p++){
		for(q=0;q<m;q++)
			printf("%f ",D[p][q]);
		printf("\n");
	}*/
//***************************theta =  D*Y *********************************************
	#pragma omp parallel for schedule(static,10) shared(m,n,D,Y,theta) private(i,j,k,sum) 
	for(i=0;i<n;i++){
		for(j=0;j<1;j++){
			sum=0.0;
			for(k=0;k<m;k++){
				sum += D[i][k]*(*(Y+k));	
			}	
			*(theta+i) = sum;
			
		}
	}


}

void loadX(float* X,int m,int n){
	int i,j;
	float p;
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){
   			scanf("%f ",&p);
			*(X+i*n+j) = p;
			
		}
	}

}
void loadY(float* Y,int m){
	int i,j;
	float p;
	for(i=0;i<m;i++){
   		scanf("%f ", &p);
		*(Y+i) = p;
		
	}
}



int main(){
	int i,j;
	 X = (float **)malloc(m * sizeof(float *));
    	for (i=0; i<m; i++)
        X[i] = (float *)malloc(n * sizeof(float));
	 theta = (float *)malloc(n * sizeof(float));	
	 Y= (float *) malloc(m * sizeof(float));
	float error=0.0;	
	 alpha = .0001;
	num_iter = 10;	


	for(i=0;i<n;i++)
		theta[i] = 2.0;
	
	srand(time(NULL));

	double sum=0.0;
	for(i=0;i<m;i++){
		sum=0.0;
		for(j=0;j<n;j++){
			if(j==0)
				X[i][j]=1.0;
			else
				 X[i][j] = (double)(rand()%1000)/1000.0;
			sum+=X[i][j];
		}
		Y[i] = sum;
	}
	/*loadX(X,m,n);
	loadY(Y,m);*/

	float time1 = omp_get_wtime();
	gradientDescent();	
	float time2 = omp_get_wtime() - time1;

	/*for(j=0;j<n;j++)
		printf("%f ",theta[j]);*/
	printf("time: %f\n",time2);

	error = finderror(X,theta,Y,m,n);
	printf("error:\t%f ",error);
	/*free(X);
	free(Y);
  	free(theta);*/
	fflush(stdout);

	return 0;
}
