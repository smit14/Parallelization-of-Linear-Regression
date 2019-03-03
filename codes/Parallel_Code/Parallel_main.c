/*****************************************************************************************************************
This is parallel code for finding value of theta in linear regression.

Input:  X: mxn matrix (Input data)
        Y: mx1 matrix (Output data)	 

Output:	Theta: nx1 matrix (parameters)

algorithm:

	X*theta = Y;
	Theta = xinv*Y;
	
	xinv = inv(Xt*X)*Xt*Y
 	
procedure:
1. find Xt              ------parallel possible
2. do   A = Xt*X	------parallel possible			<-- bottleneck (if m>>n)
3. find B = inv(A)	------one part in inherently serial	<-- bottleneck (if m~=n)
4. do   D = B*Xt	------parallel possible			<-- bottleneck (if m>>n)
5. find theta = D*Y	------parallel possible

 WE APPLY OPENMP CLAUSES HERE TO PARALLEL THESE CODES

*******************************************************************************************************************/

#include<stdio.h>
#include"omp.h"
#include<stdlib.h>
#include<time.h>
float alpha;
int num_iter;
int m=1000,n=500;			// DIMENSION OF INPUT MARTIX X
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
			sum += ((X[i][k])*theta[k]);			//ERROR = SUM{(output - actual value)^2} for all inputs
			A[i] = sum-Y[i];
			
	}
	
		
	for(i=0;i<m;i++)
		error += A[i]*A[i];
	
	error = error*100/(m*n);
	return error;

}

//********************check for invertability********************************************
int dothis(float *X,float *Y,int n,int i){
	int p,q;					//if determinant of matrix is 0 then it's not invertable
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
	#pragma omp parallel for shared(n,X,A) private(i,j) 
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			X[i][j] = *(A+i*n+j);
		}
	}
	
	//******************************Identity matrix******************************************
	#pragma omp parallel for shared(n,I) private(i,j) 	
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
		if(s==0.0){				//check for invertability
			if(!dothis(X,I,n,i)){
				return 0;
			}	
		}
		#pragma omp parallel for shared(n,X,I,i,s) private(j) 
		for(j=0;j<n;j++){					//forward process of gaussian elemination
			X[i][j]/=s;
			I[i][j]=I[i][j]/s;
		}

		#pragma omp parallel for shared(n,X,I,i) private(k,j,scalex,scalei) 
		for(j=i+1;j<n;j++){
			scalex = -X[j][i]/X[i][i];			//forward process of gaussian elemination
			scalei = scalex;
			
			for(k=0;k<n;k++){
				X[j][k] += X[i][k]*scalex;
				I[j][k] += I[i][k]*scalei;
			}		
		}
					

	}

	for(i=n-1;i>0;i--){
		#pragma omp parallel for shared(n,X,I,i) private(s,j,k) 
		for(j=i-1;j>=0;j--){
			s = -X[j][i]/X[i][i];				//backword process of gaussian elemination
						
			for(k=0;k<n;k++){
				X[j][k] += X[i][k]*s;
				I[j][k] += I[i][k]*s;
			}
		}
	} 

	#pragma omp parallel for shared(n,B,I) private(i,j) 
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
	#pragma omp parallel for shared(m,n,X,Xt) private(i,j) 
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){						//generating transpose of X
			Xt[j][i] = X[i][j];
		}
	}
	
	//*************************   Xt*X   ****************************************** 	
	#pragma omp parallel for shared(m,n,X,Xt,A) private(i,j,k,sum) 	
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			sum=0.0;
			for(k=0;k<m;k++)					//matrix multiplication Xt*X
				sum += Xt[i][k]*(X[k][j]);	
			A[i][j] = sum;
			
		}
	}

	//************************* Inverse of matrix  *********************************

	
	if(inv(A,B,n)==0){							//inverse of square matrix using gaussian elimination method
		
		return;
	}


	//***************************  B*Xt  *****************************************
	#pragma omp parallel for shared(m,n,B,Xt,D) private(i,j,k,sum) 	
	for(i=0;i<n;i++){
		for(j=0;j<m;j++){
			sum=0.0;						//matrix multiplication B*Xt
			for(k=0;k<n;k++)
				sum += B[i][k]*Xt[k][j];	
			D[i][j] = sum;
		}
	}
	
	//***************************theta =  D*Y *********************************************
	#pragma omp parallel for shared(m,n,D,Y,theta) private(i,j,k,sum) 
	for(i=0;i<n;i++){
		for(j=0;j<1;j++){
			sum=0.0;
			for(k=0;k<m;k++){					//final calculation for theta: D*Y
				sum += D[i][k]*(*(Y+k));	
			}	
			*(theta+i) = sum;
			
		}
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
	
	srand(time(NULL));

	//********************************generate data****************************
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
		float noise = (rand()%1000)/100;
		noise=noise*2-1;
		Y[i] = sum+noise;
	}
	
	//*******************************run the algorithm*************************

	float time1 = omp_get_wtime();
	gradientDescent();	
	float time2 = omp_get_wtime() - time1;

	printf("theta:\n");
	for(j=0;j<n;j++)
		printf("%f\n",theta[j]);
	printf("time: %f\n",time2);

	error = finderror(X,theta,Y,m,n);
	printf("error:\t%f ",error);
	
	return 0;
}
