#define ITMAX 2400
#define EPS 2.2204e-16
#define TOL 2.0e-4
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);
#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : -fabs(a))
#define MOV3(a,b,c, d,e,f) (a)=(d);(b)=(e);(c)=(f);
#define TALLY_MAX 100
#define RAND_RATIO 0.01


#include<stdio.h>
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include<ctype.h>
#include<cblas.h>
#include<omp.h>
#include<string.h>

// uncomment to work in single (float) precision
// ** the stimulus file must contain single precision data
// ** if the response is stored in a binary file, the file must contain single precision data
// ** output will be in single (float) precision
/*#define double float
#define cblas_dgemv cblas_sgemv
#define cblas_dgemm cblas_sgemm
#define cblas_dsymm cblas_ssymm
#define cblas_ddot cblas_sdot
#define cblas_dcopy cblas_scopy
#define cblas_dscal cblas_sscal
#define cblas_daxpy cblas_saxpy
#define cblas_dger cblas_sger
#define fabs fabsf
#define sqrt sqrtf*/

extern "C" void openblas_set_num_threads(int num_threads);

double Obj(double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, long int *slice);
void dObj(double *dcost, double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, long int *slice);

void gradCheck(double *analytic, double *empiric, double step, double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, long int *slice);

void frprmn(double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, long int *trainInd, long int *testInd, double ftol, int *iter, double *fret, double learning_rate, double momentum, int num_sched, double T, unsigned char do_sgd);
void dlinmin(double *xi, double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, int n, double *fret, long int *slice);
void mnbrak(double *xi, double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, int n, double *ax, double *bx, double *cx, double *fa, double *fb, double *fc, long int *slice);
double dbrent(double *xi, double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, int n, double ax, double bx, double cx, double tol, double *xmin, long int *slice);
double f1dim(double x, double *xi, double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, int n, long int *slice);
double df1dim(double x, double *xi, double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, int n, long int *slice);

void InitMoments(double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags);
double MinNegLikelihood(double *resp, long int Nsamples, int Nlags, long int *slice);

double sign(double x);


int main(int argc, char *argv[]) {

    unsigned char demo = 0; // run the demo implementation
    char *stimpath = argv[1]; // path to stimulus file
    char *resppath = argv[2]; // path to response file
    char *parampath = argv[3]; // path to save fit parameters and performance files
    char *paramlabel = argv[4]; // label to identify fit parameters and performance files
    unsigned char resptype = 1; // 0 for binary double; 1 for integer text
    unsigned char randomize = 0; // shuffle stimulus/response pairs to break irrelevant long-term correlations
	  unsigned char temporal_stim = 0; // use if your response contains nan; warning: higher memory usage
    unsigned char rescale_stim = 0; // 1 z-scores the stimulus features (faster fitting)

    int Ndim = 16*16; // number of stimulus dimensions
    long int Nsamples = 0; // number of stimulus samples
    int Nlags = 0; // number of temporal samples relevant to a response
    int lag_shift = 0; // number of samples delayed between stimulus presentation and response
    int rank = 4; // rank of 2nd order optimization (0 for 1st order, -1 for full rank)
    int Njack = 4; // number of jackknives to fit
    double train_fraction = 0.75; // fraction of dataset to train
    double cv_fraction = 0.25; // fraction of dataset for cross-validation (early exiting)
    double lambda = 0.0; // regularization parameter

    int Nthreads = 7; // number of threads to use for parallel processing

    unsigned char do_sgd = 0;
    double ftol = 1.0E-6;
    double momentum = 0.9;
    int num_sched = 300;
    double learning_rate = 0.2;
    double T = 0.1;

    srand(time(NULL));

    double *stim = (double*)malloc(Nsamples*Ndim * sizeof *stim);
    double *resp = (double*)malloc(Nsamples * sizeof *resp);

    int size;
    double fret;
    int iter;
    char tmp[256];
    int n = 0;
    double max = 0;
    double tmpdbl;
    long int tmpint;
    int rndNlags;
    int rnd_lag_shift;
    
    long int i, j;

    lambda = fabs(lambda);

    printf("Summary:\n");
    printf("---------------------------------\n");
    if(demo == 1) {
        printf("DEMO\n");
    }
    else {
        printf("stimpath = %s\n", stimpath);
        printf("resppath = %s\n", resppath);
        printf("parampath = %s\n", parampath);
        if(strlen(paramlabel))
            printf("paramlabel = %s\n", paramlabel);
    }
    printf("Ndim = %d\n", Ndim);
    printf("Nsamples = %ld\n", Nsamples);
    printf("lag_shift = %d\n", lag_shift);
    printf("Nlags = %d\n", Nlags);
    printf("rank = %d\n", rank);
    printf("lambda = %f\n", lambda);
    printf("randomize = %d\n", randomize);
    printf("rescale_stim = %d\n", rescale_stim);

    int max_threads = omp_get_max_threads();
    if(Nthreads > max_threads) {
        printf("ERROR: too many threads requested.\n");
        return -1;
    }

    if(Nthreads > 0) {
        omp_set_num_threads(Nthreads);
        openblas_set_num_threads(Nthreads);
    }

    printf("Nthreads[requested, <got>, max] = [%d, ", Nthreads);
    #pragma omp parallel
        Nthreads = omp_get_num_threads();
    printf("<%d>, %d]\n", Nthreads, max_threads);

    FILE *fp;

    if(demo) {
        for(i=0; i<Nsamples; i++) {
            resp[i] = 1.0*(Nsamples-i)/Nsamples;
            for(j=0; j<Ndim; j++) {
                stim[i*Ndim+j] = 1.0*(-Nsamples+i-Ndim+j)/(Nsamples+Ndim);
            }
        }
    }
    else {
        // Load stim and resp here
        if(resptype == 0)
            fp = fopen(resppath, "rb");
        else if(resptype == 1)
            fp = fopen(resppath, "r");
        if(Nsamples == 0 && resptype == 0) {
            fseek(fp, 0L, SEEK_END);
            if(resptype == 0)
                Nsamples = (int)ftell(fp)/sizeof(double);
            fseek(fp, 0L, SEEK_SET);
            printf("Nsamples found = %ld\n", Nsamples);
            resp = (double*)realloc(resp, Nsamples * sizeof *resp);
            if(Ndim != 0)
                stim = (double*)realloc(stim, Nsamples*Ndim * sizeof *stim);
        }

        if(fp == NULL) {
            printf("ERROR: fopen failed with response file \"%s\"\n", resppath);
            free(stim);
            free(resp);
            return -1;
        }

        if(resptype == 0) {
            fread(resp, sizeof *resp, Nsamples, fp);
            max = 0;
            for(i=0; i<Nsamples; i++) {
                if(resp[i] > max) {
                    max = resp[i];
                }
            }
        }
        else if(resptype == 1 && Nsamples == 0) {
            while(fgets(tmp, 30, fp)) {
                if(n+1 > Nsamples)
                    resp = (double*)realloc(resp, (n+1) * sizeof *resp);
                resp[n] = atof(tmp);
                if(resp[n] > max)
                    max = resp[n];
                n++;
                Nsamples = n;
            }
            printf("Nsamples found = %ld\n", Nsamples);
            if(Ndim != 0)
                stim = (double*)realloc(stim, Nsamples*Ndim * sizeof *stim);
        }
        else if(resptype == 1) {
            while(n < Nsamples) {
                resp[n] = atof(fgets(tmp, 256, fp));
                if(resp[n] > max)
                    max = resp[n];
                n++;
            }
        }
        fclose(fp);

        tmpdbl = 0;
        for(i=0; i<Nsamples; i++) {
            tmpdbl += resp[i];
            if(resp[i] > 0)
                resp[i] /= max;
        }
        printf("Total response = %f\n", tmpdbl);

        fp = fopen(stimpath, "rb");

        if(fp == NULL) {
            printf("ERROR: fopen failed with stimulus file \"%s\"\n", stimpath);
            free(stim);
            free(resp);
            return -1;
        }

        if(Ndim == 0) {
            fseek(fp, 0L, SEEK_END);
            Ndim = (int)ftell(fp)/(sizeof *stim);
            Ndim /= Nsamples;
            fseek(fp, 0L, SEEK_SET);
            stim = (double*)realloc(stim, Nsamples*Ndim * sizeof *stim);
            printf("Ndim found = %d\n", Ndim);
        }
        fread(stim, sizeof *stim, Nsamples*Ndim, fp);
        fclose(fp);

        if(rescale_stim) {
            // feature normalization
            double mn;
            double std;
            #pragma omp parallel for private(mn, std)
                for(i=0; i<Ndim; i++) {
                    mn = 0;
                    std = 0;
                    for(j=0; j<Nsamples; j++)
                        mn += stim[j*Ndim+i]/Nsamples;
                    for(j=0; j<Nsamples; j++)
                        std += (stim[j*Ndim+i]-mn)*(stim[j*Ndim+i]-mn)/(Nsamples-1);
                    std = sqrt(std);
                    for(j=0; j<Nsamples; j++)
                        stim[j*Ndim+i] = (stim[j*Ndim+i]-mn)/std;
            }
        }

        printf("Max response = %f\n", max);

        if(randomize || temporal_stim) {
            int *rand_perm = NULL;
            double *rand_weight = NULL;
            
            if(randomize) {
            // shuffle the data to break long-term correlations (uses more memory when Nlags > 0)
                rand_perm = (int*)malloc((Nsamples-Nlags-lag_shift) * sizeof *rand_perm);
                rand_weight = (double*)malloc((Nsamples-Nlags-lag_shift) * sizeof *rand_weight);
                #pragma omp parallel for
                    for(i=0; i<(Nsamples-Nlags-lag_shift); i++) {
                        rand_perm[i] = i;
                        rand_weight[i] = (double)rand()/RAND_MAX;
                    }

                for(i=0; i<(Nsamples-Nlags-1-lag_shift); i++) {
                    for(j=i+1; j<(Nsamples-Nlags-lag_shift); j++) {
                        if(rand_weight[i] > rand_weight[j]) {
                            tmpdbl = rand_weight[i];
                            tmpint = rand_perm[i];
                            rand_weight[i] = rand_weight[j];
                            rand_perm[i] = rand_perm[j];
                            rand_weight[j] = tmpdbl;
                            rand_perm[j] = tmpint;
                        }
                    }
                }

                free(rand_weight);
            }

            double *resp_rand = (double*)malloc((Nsamples-Nlags-lag_shift) * sizeof *resp_rand);
            double *stim_rand = (double*)malloc((Nsamples-Nlags-lag_shift)*Ndim*(Nlags+1) * sizeof *stim_rand);

            int red = 0;

            if(Nlags > 0) {
                if(randomize) {
                    for(i=0; i<(Nsamples-Nlags-lag_shift); i++) {
                        if(isnan(resp[rand_perm[i]+Nlags+lag_shift]) || isinf(resp[rand_perm[i]+Nlags+lag_shift])) {
                            red++;
                            resp_rand = (double*)realloc(resp_rand, (Nsamples-Nlags-lag_shift-red) * sizeof *resp_rand);
                            stim_rand = (double*)realloc(stim_rand, (Nsamples-Nlags-lag_shift-red)*Ndim*(Nlags+1) * sizeof *stim_rand);
                        }
                        else {
                            resp_rand[i-red] = resp[rand_perm[i]+Nlags+lag_shift];
                            memcpy(stim_rand+(i-red)*Ndim*(Nlags+1), stim+rand_perm[i]*Ndim, Ndim*(Nlags+1)*sizeof(double));
                        }
                    }
                }
                else {
                    for(i=0; i<(Nsamples-Nlags-lag_shift); i++) {
                        if(isnan(resp[i+Nlags+lag_shift]) || isinf(resp[i+Nlags+lag_shift])) {
                            red++;
                            resp_rand = (double*)realloc(resp_rand, (Nsamples-Nlags-lag_shift-red) * sizeof *resp_rand);
                            stim_rand = (double*)realloc(stim_rand, (Nsamples-Nlags-lag_shift-red)*Ndim*(Nlags+1) * sizeof *stim_rand);
                        }
                        else {
                            resp_rand[i-red] = resp[i+Nlags+lag_shift];
                            memcpy(stim_rand+(i-red)*Ndim*(Nlags+1), stim+i*Ndim, Ndim*(Nlags+1)*sizeof(double));
                        }
                    }                 
                }
            }
            else {
                if(randomize) {
                    for(i=0; i<(Nsamples-lag_shift); i++) {
                        if(isnan(resp[rand_perm[i]+Nlags+lag_shift]) || isinf(resp[rand_perm[i]+Nlags+lag_shift])) {
                            red++;
                            resp_rand = (double*)realloc(resp_rand, (Nsamples-lag_shift-red) * sizeof *resp_rand);
                            stim_rand = (double*)realloc(stim_rand, (Nsamples-lag_shift-red)*Ndim * sizeof *stim_rand);
                        }
                        else {
                            resp_rand[i-red] = resp[rand_perm[i]+lag_shift];
                            memcpy(stim_rand+(i-red)*Ndim, stim+rand_perm[i]*Ndim, Ndim*sizeof(double));
                        }
                    }
                }
                else {
                    for(i=0; i<(Nsamples-lag_shift); i++) {
                        if(isnan(resp[i+Nlags+lag_shift]) || isinf(resp[i+Nlags+lag_shift])) {
                            red++;
                            resp_rand = (double*)realloc(resp_rand, (Nsamples-lag_shift-red) * sizeof *resp_rand);
                            stim_rand = (double*)realloc(stim_rand, (Nsamples-lag_shift-red)*Ndim * sizeof *stim_rand);
                        }
                        else {
                            resp_rand[i-red] = resp[i+lag_shift];
                            memcpy(stim_rand+(i-red)*Ndim, stim+i*Ndim, Ndim*sizeof(double));
                        }
                    }
                }
            }
            if(red) {
                tmpdbl = 0;
                for(i=0; i<Nsamples-lag_shift-Nlags-red; i++)
                    tmpdbl += resp_rand[i];
                printf("Adjusted total response = %f\n", tmpdbl*max);
            }

            free(rand_perm);
            free(resp);
            free(stim);

            resp = resp_rand;
            stim = stim_rand;
            resp_rand = NULL;
            stim_rand = NULL;

            rndNlags = Nlags;
            rnd_lag_shift = lag_shift;
            Nsamples -= Nlags+lag_shift+red;
            Ndim *= (Nlags+1);
            lag_shift = 0;
            Nlags = 0;
        }

        // put max at 0.5 to make the max approx. linear
        //for(i=0; i<Nsamples; i++)
        //    resp[i] *= 0.5;
    }

    if(rank < 0) {
        rank = Ndim*(Nlags+1);
        printf("adjusted rank = %d\n", rank);
    }

    Nsamples -= lag_shift;

    double *p;

    if(rank > Ndim*(Nlags+1)/2) {
        size = 1+Ndim*(Nlags+1)+Ndim*(Nlags+1)*(Ndim*(Nlags+1)+1)/2;
        rank = Ndim*(Nlags+1);
    }
    else {
        if(rank == 0)
            size = 1+Ndim*(Nlags+1);
        else
            size = 1+(rank+1)*(Ndim*(Nlags+1)+1);
    }

    p = (double*)malloc(size * sizeof *p);
    // random initialization
    if(rank > Ndim*(Nlags+1)/2)
        InitMoments(p, rank, stim, Ndim, resp+lag_shift, Nsamples, Nlags);
    else
        InitMoments(p, 0, stim, Ndim, resp+lag_shift, Nsamples, Nlags);
    if(p[0] != 0) {
        p[0] = log(1.0/p[0]-1.0);
        for(i=1; i<size; i++)
            p[i] = 0.001*(2.0*rand()/RAND_MAX-1.0);
    }
    else {
        for(i=0; i<size; i++)
            p[i] = 0.001*(2.0*rand()/RAND_MAX-1.0);
    }
    
    // initialize by moments
    /*if(rank == 0 || rank > Ndim*(Nlags+1)/2)
        InitMoments(p, rank, stim, Ndim, resp+lag_shift, Nsamples, Nlags);
    else {        
        for(i=1+Ndim*(Nlags+1); i<size; i++) {
            p[i] = 0.001*(2.0*rand()/RAND_MAX-1.0);
        }
    }*/

    // Gradient checking
    /*double *analytic = (double*)malloc(size * sizeof *analytic);
    double *empiric = (double*)malloc(size * sizeof *empiric);
    long int slice[2];
    slice[0] = 0;
    slice[1] = 50-Nlags;

    gradCheck(analytic, empiric, 1E-6, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice);

    printf("Analytic : Empiric\n");
    printf("a = %f  :  %f\n", analytic[0], empiric[0]);
    for(i=0; i<Ndim*(Nlags+1); i++)
        printf("h[%d] = %f  :  %f\n", i+1, analytic[i+1], empiric[i+1]);
    if(rank > Ndim*(Nlags+1)/2) {
        for(i=0; i<Ndim*(Nlags+1); i++)
            for(j=0; j<Ndim*(Nlags+1)-i; j++ )
                printf("J[%d][%d] = %f  :  %f\n", i, j, analytic[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i+j], empiric[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i+j]);
    }
    else if(rank > 0) {
        for(i=0; i<rank; i++)
            for(j=0; j<Ndim*(Nlags+1); j++)
                printf("V[%d][%d] = %f  :  %f\n", j, i, analytic[1+(i+1)*Ndim*(Nlags+1)+j], empiric[1+(i+1)*Ndim*(Nlags+1)+j]);
        for(i=0; i<rank; i++)
            printf("U[%d][%d] = %f  :  %f\n", i, i, analytic[1+(rank+1)*Ndim*(Nlags+1)+i], empiric[1+(rank+1)*Ndim*(Nlags+1)+i]);
    }

    free(analytic);
    free(empiric);
    free(stim);
    free(resp);
    free(p);

    return 0;*/

    printf("Stimuli and response loaded\n");
    if(Nsamples-Nlags >= 10) {
        printf("First 10 response bins (divided by max): ");
        for(i=0; i<10; i++) {
            printf("%lf ", resp[i+Nlags+lag_shift]);
        }
        printf("\n");
    }

    printf("Njack = %d\n", Njack);

    if(cv_fraction <= 1.0/(Nsamples-Nlags) || cv_fraction >= 1) {
        printf("ERROR: cv_fraction must be < 1 and > %f.\n", 1.0/Nsamples);
        return -1;
    }
    if(train_fraction <= 1.0*Nlags/(Nsamples-Nlags) || train_fraction >= 1) {
        printf("ERROR: train_fraction must be < 1 and > %f.\n", 1.0/Nsamples);
        return -1;
    }
    if(cv_fraction+train_fraction > 1) {
        printf("ERROR: train_fraction and cv_fraction must add up to <= 1.\n");
        return -1;
    }

    long int Ncv = (int)((Nsamples-Nlags)*cv_fraction);
    long int Ntrain = (int)((Nsamples-Nlags)*train_fraction);
    long int Ntest = Nsamples-Nlags-Ncv-Ntrain;
    long int Nstep = ((Nsamples-Nlags)/Njack);

    if(Nlags > Nsamples-Nlags || Nlags > Nsamples-Nlags) {
        printf("ERROR: Nlags must be <= Nsamples-Nlags\n");
        return -1;
    }

    printf("Ntrain = %ld\n", Ntrain);
    printf("Ncv = %ld\n", Ncv);
    printf("---------------------------------\n\n");
    printf("Optimization:\n");
    printf("---------------------------------\n");

    long int trainInd[2];
    long int cvInd[2];
    long int testInd[2];
    double ftest;
    double X;
    char ppf[256];
    int jack;

    if(randomize)
        sprintf(ppf, "%s/predictive_power%s_r%d_l%d_n%d.dat", parampath, paramlabel, rank, rnd_lag_shift, rndNlags);
    else
        sprintf(ppf, "%s/predictive_power%s_r%d_l%d_n%d.dat", parampath, paramlabel, rank, lag_shift, Nlags);
    fp = fopen(ppf, "wb");
    fclose(fp);

    for(jack=0; jack<Njack; jack++) {

        trainInd[0] = (jack*Nstep)%(Nsamples-Nlags);
        trainInd[1] = (trainInd[0]+Ntrain-1)%(Nsamples-Nlags)+1;
        cvInd[0] = (trainInd[1])%(Nsamples-Nlags);
        cvInd[1] = (cvInd[0]+Ncv-1)%(Nsamples-Nlags)+1;
        testInd[0] = (cvInd[1])%(Nsamples-Nlags);
        testInd[1] = (testInd[0]+Ntest-1)%(Nsamples-Nlags)+1;

        printf("jackknife = %d;  train=[%ld, %ld];  cv=[%ld, %ld];  test=[%ld, %ld]\n", jack, trainInd[0], trainInd[1], cvInd[0], cvInd[1], testInd[0], testInd[1]);

        frprmn(p, rank, stim, Ndim, resp+lag_shift, Nsamples, Nlags, lambda, trainInd, cvInd, ftol, &iter, &fret, learning_rate, momentum, num_sched, T, do_sgd);

        ftest = Obj(p, rank, stim, Ndim, resp+lag_shift, Nsamples, Nlags, 0.0, testInd);
        X = (log(2)-ftest)/(log(2)-MinNegLikelihood(resp+lag_shift, Nsamples, Nlags, testInd));

        printf("predictive power: X = %f (ftest = %f)\n", X, ftest);

        if(demo == 0) {
            printf("Writing to file\n");
            if(randomize)
                sprintf(tmp, "%s/params%s_r%d_l%d_n%d_j%d.dat", parampath, paramlabel, rank, rnd_lag_shift, rndNlags, jack);
            else
                sprintf(tmp, "%s/params%s_r%d_l%d_n%d_j%d.dat", parampath, paramlabel, rank, lag_shift, Nlags, jack);
            fp = fopen(tmp, "wb");
            fclose(fp);
            fp = fopen(tmp, "ab");
            fwrite(p, sizeof(double), 1+Ndim*(Nlags+1), fp);
            if(rank > Ndim*(Nlags+1)/2) {
                for(i=0; i<Ndim*(Nlags+1); i++) {
                    for(j=0; j<i; j++)
                        fwrite(p+1+Ndim*(Nlags+1)+j*Ndim*(Nlags+1)-j*(j+1)/2+i, sizeof(double), 1, fp);
                    for(j=0; j<Ndim*(Nlags+1)-i; j++)
                        fwrite(p+1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i+j, sizeof(double), 1, fp);
                }
            }
            else if(rank > 0) {
                fwrite(p+1+Ndim*(Nlags+1), sizeof(double), rank*(Ndim*(Nlags+1)+1), fp);
            }
            fclose(fp);
            fp = fopen(ppf, "ab");
            fwrite(&X, sizeof(double), 1, fp);
            fwrite(&ftest, sizeof(double), 1, fp);
            fclose(fp);
            printf("Done.\n");
        }

        if(jack < Njack-1) {
            // random initialization
            if(rank > Ndim*(Nlags+1)/2)
                InitMoments(p, rank, stim, Ndim, resp+lag_shift, Nsamples, Nlags);
            else
                InitMoments(p, 0, stim, Ndim, resp+lag_shift, Nsamples, Nlags);
            if(p[0] != 0) {
                p[0] = log(1.0/p[0]-1.0);
                for(i=1; i<size; i++)
                    p[i] = 0.001*(2.0*rand()/RAND_MAX-1.0);
            }
            else {
                for(i=0; i<size; i++)
                    p[i] = 0.001*(2.0*rand()/RAND_MAX-1.0);
            }
            
            // initialize by moments
            /*if(rank == 0 || rank > Ndim*(Nlags+1)/2)
                InitMoments(p, rank, stim, Ndim, resp+lag_shift, Nsamples, Nlags);
            else
                for(i=0; i<size; i++) {
                    p[i] = 0.05*(2.0*rand()/RAND_MAX-1.0);
                }*/
        }
    }

    printf("-------------------------------\n");

    printf("Complete.\n");

    free(stim);
    free(resp);
    free(p);

    return 0;
}



void InitMoments(double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags) {

    double a = 0;
    double *h = (double*)calloc(Ndim*(Nlags+1), sizeof *h);
    double *J;
    double *TMP;
    long int Nsamp = Nsamples-Nlags;
    long int i, j, t;
    if(rank > 0) {
        J = (double*)calloc(Ndim*(Nlags+1)*Ndim*(Nlags+1), sizeof *J);
        TMP = (double*)malloc(((Nsamp-1)/(Nlags+1)+1)*Ndim*(Nlags+1) * sizeof *TMP);
    }

    for(i=0; i<=Nlags; i++) {
        if(Nsamp-i < 1)
            break;
        #pragma omp parallel for reduction(+:a)
            for(t=0; t<(Nsamp-i-1)/(Nlags+1)+1; t++)
                a += resp[i+t*(Nlags+1)+Nlags];
        cblas_dgemv(CblasRowMajor, CblasTrans, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, stim+i*Ndim, Ndim*(Nlags+1), resp+i, Nlags+1, 1, h, 1);
        //cblas_sgemv(CblasRowMajor, CblasTrans, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, stim+i*Ndim, Ndim*(Nlags+1), resp+i, Nlags+1, 1, h, 1);
        if(rank > 0) {
            #pragma omp parallel for
                for(t=0; t<(Nsamp-i-1)/(Nlags+1)+1; t++)
                    for(j=0; j<Ndim*(Nlags+1); j++)
                        TMP[t*Ndim*(Nlags+1)+j] = resp[i+t*(Nlags+1)+Nlags]*stim[i*Ndim+t*Ndim*(Nlags+1)+j];
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Ndim*(Nlags+1), Ndim*(Nlags+1), (Nsamp-i-1)/(Nlags+1)+1, 1, TMP, Ndim*(Nlags+1), stim+i*Ndim, Ndim*(Nlags+1), 1, J, Ndim*(Nlags+1));
            //cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Ndim*(Nlags+1), Ndim*(Nlags+1), (Nsamp-i-1)/(Nlags+1)+1, 1, TMP, Ndim*(Nlags+1), stim+i*Ndim, Ndim*(Nlags+1), 1, J, Ndim*(Nlags+1));
        }
    }

    p[0] = a/Nsamp;
    for(i=0; i<Ndim*(Nlags+1); i++) {
        p[1+i] = h[i]/Nsamp;
        if(rank > 0) {
            for(j=0; j<Ndim*(Nlags+1)-i; j++) {
                p[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i+j] = J[i*Ndim*(Nlags+1)+i+j]/Nsamp;
            }
        }
    }

    free(h);
    if(rank > 0) {
        free(J);
        free(TMP);
    }

    return;
}


double MinNegLikelihood(double *resp, long int Nsamples, int Nlags, long int *slice) {

    double nl = 0;
    int tmpind;
    long int i;

    if(slice[0] > slice[1]) {
        tmpind = slice[0];
        slice[0] = 0;

        nl = MinNegLikelihood(resp, Nsamples, Nlags, slice)*slice[1];

        slice[0] = tmpind;
        tmpind = slice[1];
        slice[1] = Nsamples-Nlags;

        nl += MinNegLikelihood(resp, Nsamples, Nlags, slice)*(Nsamples-Nlags-slice[0]);

        slice[1] = tmpind;

        return nl/(Nsamples-Nlags-slice[0]+slice[1]);
    }
    else if(slice[0] == slice[1]) {
        tmpind = slice[0];
        slice[0] = 0;
        slice[1] = Nsamples-Nlags;

        nl = MinNegLikelihood(resp, Nsamples, Nlags, slice);

        slice[0] = slice[1] = tmpind;

        return nl;
    }

    #pragma omp parallel for reduction(+:nl)
    for(i=slice[0]; i<slice[1]; i++)
        nl += -resp[i]*log(resp[i]+EPS) - (1-resp[i])*log(1-resp[i]+EPS);

    return nl/(slice[1]-slice[0]);

}



double Obj(double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, long int *slice) {
    // slice[0] = range(0, Nsamples-Nlags-1)
    // slice[1] = range(1, Nsamples-Nlags)

    double tmp;
    int Nsamp;
    double cost = 0;
    int tmpind;
    long int i, j, t;

    if(slice[0] < slice[1])
        Nsamp = slice[1]-slice[0];
    else if(slice[0] > slice[1]) {
        Nsamp = Nsamples-Nlags-slice[0]+slice[1];

        tmpind = slice[0];
        slice[0] = 0;
        cost += Obj(p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice)*slice[1];

        slice[0] = tmpind;
        tmpind = slice[1];
        slice[1] = Nsamples-Nlags;
        cost += Obj(p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice)*(Nsamp-tmpind);
        slice[1] = tmpind;

        return cost/Nsamp;
    }
    else {
        Nsamp = Nsamples-Nlags;

        tmpind = slice[0];
        slice[0] = 0;
        slice[1] = Nsamp;

        cost = Obj(p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice);

        slice[0] = slice[1] = tmpind;

        return cost;
    }

    double a = p[0];
    double *h = (double*)malloc(Ndim*(Nlags+1) * sizeof *h);
    double *J;
    double *TMP;
    double *L;
    double *U;
    double *VT;
    double *UVT;
    if(rank > 0) {
        J = (double*)malloc(Ndim*(Nlags+1)*Ndim*(Nlags+1) * sizeof *J);
        TMP = (double*)malloc(((Nsamp-1)/(Nlags+1)+1)*Ndim*(Nlags+1) * sizeof *TMP);
        if(rank <= Ndim*(Nlags+1)/2) {
            VT = (double*)malloc(rank*Ndim*(Nlags+1) * sizeof *VT);
            UVT = (double*)malloc(rank*Ndim*(Nlags+1) * sizeof *UVT);
            U = (double*)calloc(rank*rank, sizeof *U);
            for(i=0; i<rank; i++) {
                U[i*rank+i] = p[1+Ndim*(Nlags+1)+rank*Ndim*(Nlags+1)+i];
                for(j=0; j<Ndim*(Nlags+1); j++) {
                    VT[j*rank+i] = p[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)+j];
                }
            }
        }
    }
    L = (double*)malloc(((Nsamp-1)/(Nlags+1)+1) * sizeof *L);

    for(i=0; i<Ndim*(Nlags+1); i++) {
        h[i] = p[i+1];
        if(rank > Ndim*(Nlags+1)/2) {
            for(j=0; j<Ndim*(Nlags+1)-i; j++) {
                J[i*Ndim*(Nlags+1)+i+j] = p[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i+j];
                J[j*Ndim*(Nlags+1)+i*Ndim*(Nlags+1)+i] = J[i*Ndim*(Nlags+1)+i+j];
            }
        }
    }
    
    if(rank > 0 && rank <= Ndim*(Nlags+1)/2) {
        cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, rank, Ndim*(Nlags+1), 1, U, rank, p+1+Ndim*(Nlags+1), Ndim*(Nlags+1), 0, UVT, Ndim*(Nlags+1));
        //cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, rank, Ndim*(Nlags+1), 1, U, rank, p+1+Ndim*(Nlags+1), Ndim*(Nlags+1), 0, UVT, Ndim*(Nlags+1));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Ndim*(Nlags+1), Ndim*(Nlags+1), rank, 1, VT, rank, UVT, Ndim*(Nlags+1), 0, J, Ndim*(Nlags+1));
        //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Ndim*(Nlags+1), Ndim*(Nlags+1), rank, 1, VT, rank, UVT, Ndim*(Nlags+1), 0, J, Ndim*(Nlags+1));
    }

    for(i=0; i<=Nlags; i++) {
        if(Nsamp-i < 1)
            break;
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), h, 1, 0, L, 1);
        //cblas_sgemv(CblasRowMajor, CblasNoTrans, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), h, 1, 0, L, 1);
        if(rank > 0) {
            cblas_dsymm(CblasRowMajor, CblasRight, CblasLower, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, J, Ndim*(Nlags+1), stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), 0, TMP, Ndim*(Nlags+1));
            //cblas_ssymm(CblasRowMajor, CblasRight, CblasLower, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, J, Ndim*(Nlags+1), stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), 0, TMP, Ndim*(Nlags+1));
            for(t=0; t<(Nsamp-i-1)/(Nlags+1)+1; t++) {
                tmp = 1.0/(1.0+exp(a + L[t] + cblas_ddot(Ndim*(Nlags+1), stim+(slice[0]+i)*Ndim+t*Ndim*(Nlags+1), 1, TMP+t*Ndim*(Nlags+1), 1)));
                //tmp = 1.0/(1.0+exp(a + L[t] + cblas_sdot(Ndim*(Nlags+1), stim+(slice[0]+i)*Ndim+t*Ndim*(Nlags+1), 1, TMP+t*Ndim*(Nlags+1), 1)));
                cost -= resp[slice[0]+i+t*(Nlags+1)+Nlags]*log(tmp+EPS) + (1-resp[slice[0]+i+t*(Nlags+1)+Nlags])*log(1-tmp+EPS);
            }
        }
        else {
            for(t=0; t<(Nsamp-i-1)/(Nlags+1)+1; t++) {
                tmp = 1.0/(1.0+exp(a + L[t]));
                cost -= resp[slice[0]+i+t*(Nlags+1)+Nlags]*log(tmp+EPS) + (1-resp[slice[0]+i+t*(Nlags+1)+Nlags])*log(1-tmp+EPS);
            }
        }
    }

    cost /= Nsamp;

    if(rank > Ndim*(Nlags+1)/2) {
        for(i=0; i<Ndim*(Nlags+1); i++) {
            // cost += lambda*p[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i]*p[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i]/2;
            // cost += lambda*sqrt(fabs(p[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i]))/2;
            cost += lambda*fabs(p[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i])/2;
            #pragma omp parallel for reduction(+:cost)
                for(j=1; j<Ndim*(Nlags+1)-i; j++)
                    cost += lambda*fabs(p[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i+j]);
                    // cost += lambda*sqrt(fabs(p[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i+j]));
                    // cost += lambda*p[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i+j]*p[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i+j];
        }
    }
    if(rank > 0 && rank <= Ndim*(Nlags+1)/2) {
        for(i=0; i<rank; i++) {
            cost += lambda*p[1+(rank+1)*Ndim*(Nlags+1)+i]*p[1+(rank+1)*Ndim*(Nlags+1)+i]/2;
            for(j=0; j<Ndim*(Nlags+1); j++) {
                cost += lambda*p[1+(i+1)*Ndim*(Nlags+1)+j]*p[1+(i+1)*Ndim*(Nlags+1)+j]/2;
            }
        }
    }

    free(h);
    free(L);
    if(rank > 0) {
        free(J);
        free(TMP);
        if(rank <= Ndim*(Nlags+1)/2) {
            free(U);
            free(VT);
            free(UVT);
        }
    }

    return cost;

}



void dObj(double *dcost, double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, long int *slice) {

    int size;
    int Nsamp;
    double tmp;
    int tmpind;
    long int i, j, k, l, t;

    if(slice[0] < slice[1])
        Nsamp = slice[1]-slice[0];
    else if(slice[0] > slice[1]) {
        double *dcosttemp;
        if(rank > Ndim*(Nlags+1)/2) {
            dcosttemp = (double*)malloc((1+Ndim*(Nlags+1)+Ndim*(Nlags+1)*(Ndim*(Nlags+1)+1)/2) * sizeof *dcosttemp);
            cblas_dcopy(1+Ndim*(Nlags+1)+Ndim*(Nlags+1)*(Ndim*(Nlags+1)+1)/2, dcost, 1, dcosttemp, 1);
            //cblas_scopy(1+Ndim*(Nlags+1)+Ndim*(Nlags+1)*(Ndim*(Nlags+1)+1)/2, dcost, 1, dcosttemp, 1);
        }
        else if(rank > 0) {
            dcosttemp = (double*)malloc((1+Ndim*(Nlags+1)+rank*(Ndim*(Nlags+1)+1)) * sizeof *dcosttemp);
            cblas_dcopy(1+Ndim*(Nlags+1)+rank*(Ndim*(Nlags+1)+1), dcost, 1, dcosttemp, 1);
            //cblas_scopy(1+Ndim*(Nlags+1)+rank*(Ndim*(Nlags+1)+1), dcost, 1, dcosttemp, 1);
        }
        else {
            dcosttemp = (double*)malloc((1+Ndim*(Nlags+1)) * sizeof *dcosttemp);
            cblas_dcopy(1+Ndim*(Nlags+1), dcost, 1, dcosttemp, 1);
            //cblas_scopy(1+Ndim*(Nlags+1), dcost, 1, dcosttemp, 1);
        }
        Nsamp = Nsamples-Nlags-slice[0]+slice[1];

        tmpind = slice[0];
        slice[0] = 0;
        if(slice[1] > 0)
            dObj(dcosttemp, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice);

        slice[0] = tmpind;
        tmpind = slice[1];
        slice[1] = Nsamples-Nlags;
        if(slice[0] < Nsamples-Nlags)
            dObj(dcost, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice);

        slice[1] = tmpind;

        if(rank > Ndim*(Nlags+1)/2) {
            cblas_dscal(1+Ndim*(Nlags+1)+Ndim*(Nlags+1)*(Ndim*(Nlags+1)+1)/2, 1.0*(Nsamp-slice[1])/Nsamp, dcost, 1);
            //cblas_sscal(1+Ndim*(Nlags+1)+Ndim*(Nlags+1)*(Ndim*(Nlags+1)+1)/2, 1.0*(Nsamp-slice[1])/Nsamp, dcost, 1);
            cblas_daxpy(1+Ndim*(Nlags+1)+Ndim*(Nlags+1)*(Ndim*(Nlags+1)+1)/2, 1.0*slice[1]/Nsamp, dcosttemp, 1, dcost, 1);
            //cblas_saxpy(1+Ndim*(Nlags+1)+Ndim*(Nlags+1)*(Ndim*(Nlags+1)+1)/2, 1.0*slice[1]/Nsamp, dcosttemp, 1, dcost, 1);
        }
        else if(rank > 0) {
            cblas_dscal(1+Ndim*(Nlags+1)+rank*(Ndim*(Nlags+1)+1), 1.0*(Nsamp-slice[1])/Nsamp, dcost, 1);
            //cblas_sscal(1+Ndim*(Nlags+1)+rank*(Ndim*(Nlags+1)+1), 1.0*(Nsamp-slice[1])/Nsamp, dcost, 1);
            cblas_daxpy(1+Ndim*(Nlags+1)+rank*(Ndim*(Nlags+1)+1), 1.0*slice[1]/Nsamp, dcosttemp, 1, dcost, 1);
            //cblas_saxpy(1+Ndim*(Nlags+1)+rank*(Ndim*(Nlags+1)+1), 1.0*slice[1]/Nsamp, dcosttemp, 1, dcost, 1);
        }
        else {
            cblas_dscal(1+Ndim*(Nlags+1), 1.0*(Nsamp-slice[1])/Nsamp, dcost, 1);
            //cblas_sscal(1+Ndim*(Nlags+1), 1.0*(Nsamp-slice[1])/Nsamp, dcost, 1);
            cblas_daxpy(1+Ndim*(Nlags+1), 1.0*slice[1]/Nsamp, dcosttemp, 1, dcost, 1);
            //cblas_saxpy(1+Ndim*(Nlags+1), 1.0*slice[1]/Nsamp, dcosttemp, 1, dcost, 1);
        }

        free(dcosttemp);

        return;
    }
    else {
        Nsamp = Nsamples-Nlags;

        tmp = slice[0];
        slice[0] = 0;
        slice[1] = Nsamp;

        dObj(dcost, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice);

        slice[0] = slice[1] = tmp;

        return;
    }

    double a = p[0];
    double *h = (double*)malloc(Ndim*(Nlags+1) * sizeof *h);
    double *J;
    double *U;
    double *VT;
    double *UVT;
    double *VV;
    double *TMP;
    double *L;
    double *COV;
    if(rank > 0) {
        J = (double*)malloc(Ndim*(Nlags+1)*Ndim*(Nlags+1) * sizeof *J);
        TMP = (double*)malloc(((Nsamp-1)/(Nlags+1)+1)*Ndim*(Nlags+1) * sizeof *TMP);
        COV = (double*)calloc(Ndim*(Nlags+1)*Ndim*(Nlags+1), sizeof *COV);
        if(rank <= Ndim*(Nlags+1)/2) {
            size = 1+Ndim*(Nlags+1)+rank*(Ndim*(Nlags+1)+1);
            VV = (double*)calloc(Ndim*(Nlags+1)*Ndim*(Nlags+1), sizeof *VV);
            VT = (double*)malloc(rank*Ndim*(Nlags+1) * sizeof *VT);
            UVT = (double*)malloc(rank*Ndim*(Nlags+1) * sizeof *UVT);
            U = (double*)calloc(rank*rank, sizeof *U);
            for(i=0; i<rank; i++) {
                U[i*rank+i] = p[1+Ndim*(Nlags+1)+rank*Ndim*(Nlags+1)+i];
                dcost[1+(rank+1)*Ndim*(Nlags+1)+i] = 0;
                for(j=0; j<Ndim*(Nlags+1); j++) {
                    dcost[1+(i+1)*Ndim*(Nlags+1)+j] = 0;
                    VT[j*rank+i] = p[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)+j];
                }
            }
        }
        else
            size = 1+Ndim*(Nlags+1)+Ndim*(Nlags+1)*(Ndim*(Nlags+1)+1)/2;
    }
    else
        size = 1+Ndim*(Nlags+1);
    L = (double*)malloc(((Nsamp-1)/(Nlags+1)+1) * sizeof *L);

    dcost[0] = 0;
    for(i=0; i<Ndim*(Nlags+1); i++) {
        h[i] = p[i+1];
        dcost[1+i] = 0;
        if(rank > Ndim*(Nlags+1)/2) {
            for(j=0; j<Ndim*(Nlags+1)-i; j++) {
                J[i*Ndim*(Nlags+1)+i+j] = p[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i+j];
                J[j*Ndim*(Nlags+1)+i*Ndim*(Nlags+1)+i] = J[i*Ndim*(Nlags+1)+i+j];
                dcost[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i+j] = 0;
            }
        }
    }
    if(rank > 0 && rank <= Ndim*(Nlags+1)/2) {
        cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, rank, Ndim*(Nlags+1), 1, U, rank, p+1+Ndim*(Nlags+1), Ndim*(Nlags+1), 0, UVT, Ndim*(Nlags+1));
        //cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, rank, Ndim*(Nlags+1), 1, U, rank, p+1+Ndim*(Nlags+1), Ndim*(Nlags+1), 0, UVT, Ndim*(Nlags+1));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Ndim*(Nlags+1), Ndim*(Nlags+1), rank, 1, VT, rank, UVT, Ndim*(Nlags+1), 0, J, Ndim*(Nlags+1));
        //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Ndim*(Nlags+1), Ndim*(Nlags+1), rank, 1, VT, rank, UVT, Ndim*(Nlags+1), 0, J, Ndim*(Nlags+1));
    }

    for(i=0; i<=Nlags; i++) {
        if(Nsamp-i < 1)
            break;
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), h, 1, 0, L, 1);
        //cblas_sgemv(CblasRowMajor, CblasNoTrans, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), h, 1, 0, L, 1);
        if(rank > 0) {
            cblas_dsymm(CblasRowMajor, CblasRight, CblasLower, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, J, Ndim*(Nlags+1), stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), 0, TMP, Ndim*(Nlags+1));
            //cblas_ssymm(CblasRowMajor, CblasRight, CblasLower, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, J, Ndim*(Nlags+1), stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), 0, TMP, Ndim*(Nlags+1));
            for(t=0; t<(Nsamp-i-1)/(Nlags+1)+1; t++) {
                L[t] = resp[slice[0]+i+t*(Nlags+1)+Nlags] - 1.0/(1.0+exp(a + L[t] + cblas_ddot(Ndim*(Nlags+1), stim+(slice[0]+i)*Ndim+t*Ndim*(Nlags+1), 1, TMP+t*Ndim*(Nlags+1), 1)));
                //L[t] = resp[slice[0]+i+t*(Nlags+1)+Nlags] - 1.0/(1.0+exp(a + L[t] + cblas_sdot(Ndim*(Nlags+1), stim+(slice[0]+i)*Ndim+t*Ndim*(Nlags+1), 1, TMP+t*Ndim*(Nlags+1), 1)));
                dcost[0] += L[t];
            }
            cblas_dgemv(CblasRowMajor, CblasTrans, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), L, 1, 1, dcost+1, 1);
            //cblas_sgemv(CblasRowMajor, CblasTrans, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), L, 1, 1, dcost+1, 1);
            #pragma omp parallel for
            for(t=0; t<(Nsamp-i-1)/(Nlags+1)+1; t++)
                for(j=0; j<Ndim*(Nlags+1); j++)
                    TMP[t*Ndim*(Nlags+1)+j] = L[t]*stim[(slice[0]+i)*Ndim+t*Ndim*(Nlags+1)+j];
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Ndim*(Nlags+1), Ndim*(Nlags+1), (Nsamp-i-1)/(Nlags+1)+1, 1, TMP, Ndim*(Nlags+1), stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), 0, COV, Ndim*(Nlags+1));
            //cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Ndim*(Nlags+1), Ndim*(Nlags+1), (Nsamp-i-1)/(Nlags+1)+1, 1, TMP, Ndim*(Nlags+1), stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), 0, COV, Ndim*(Nlags+1));
            if(rank > Ndim*(Nlags+1)/2) {
                for(j=0; j<Ndim*(Nlags+1); j++) {
                    dcost[1+(j+1)*Ndim*(Nlags+1)-j*(j+1)/2+j] += COV[j*Ndim*(Nlags+1)+j];
                    for(k=1; k<Ndim*(Nlags+1)-j; k++) {
                        dcost[1+(j+1)*Ndim*(Nlags+1)-j*(j+1)/2+j+k] += 2*COV[j*Ndim*(Nlags+1)+j+k];
                    }
                }
            }
            else {
                for(j=0; j<rank; j++) {
                    cblas_dger(CblasRowMajor, Ndim*(Nlags+1), Ndim*(Nlags+1), 1, p+1+(j+1)*Ndim*(Nlags+1), 1, p+1+(j+1)*Ndim*(Nlags+1), 1, VV, Ndim*(Nlags+1));
                    //cblas_sger(CblasRowMajor, Ndim*(Nlags+1), Ndim*(Nlags+1), 1, p+1+(j+1)*Ndim*(Nlags+1), 1, p+1+(j+1)*Ndim*(Nlags+1), 1, VV, Ndim*(Nlags+1));
                    for(k=0; k<Ndim*(Nlags+1); k++) {
                        tmp = cblas_ddot(Ndim*(Nlags+1), COV+k, Ndim*(Nlags+1), p+1+(j+1)*Ndim*(Nlags+1), 1);
                        //tmp = cblas_sdot(Ndim*(Nlags+1), COV+k, Ndim*(Nlags+1), p+1+(j+1)*Ndim*(Nlags+1), 1);
                        dcost[1+Ndim*(Nlags+1)+j*Ndim*(Nlags+1)+k] += 2*tmp*U[j*rank+j];
                        dcost[1+Ndim*(Nlags+1)+rank*Ndim*(Nlags+1)+j] += VV[k*Ndim*(Nlags+1)+k]*COV[k*Ndim*(Nlags+1)+k];
                        VV[k*Ndim*(Nlags+1)+k] = 0;
                        for(l=k+1; l<Ndim*(Nlags+1); l++) {
                            dcost[1+Ndim*(Nlags+1)+rank*Ndim*(Nlags+1)+j] += 2*VV[k*Ndim*(Nlags+1)+l]*COV[k*Ndim*(Nlags+1)+l];
                            VV[k*Ndim*(Nlags+1)+l] = 0;
                        }
                    }
                }
            }
        }
        else {
            for(t=0; t<(Nsamp-i-1)/(Nlags+1)+1; t++) {
                L[t] = resp[slice[0]+i+t*(Nlags+1)+Nlags] - 1.0/(1.0+exp(a + L[t]));
                dcost[0] += L[t];
            }
            cblas_dgemv(CblasRowMajor, CblasTrans, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), L, 1, 1, dcost+1, 1);
            //cblas_sgemv(CblasRowMajor, CblasTrans, (Nsamp-i-1)/(Nlags+1)+1, Ndim*(Nlags+1), 1, stim+(slice[0]+i)*Ndim, Ndim*(Nlags+1), L, 1, 1, dcost+1, 1);
        }
    }

    dcost[0] /= Nsamp;
    for(i=0; i<Ndim*(Nlags+1); i++) {
        dcost[i+1] /= Nsamp;
        if(rank > Ndim*(Nlags+1)/2) {
            dcost[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i] /= Nsamp;
            dcost[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i] += lambda*sign(p[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i])/2;
            // dcost[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i] += lambda*p[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i];
            // dcost[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i] -= lambda*sign(p[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i])/sqrt(fabs(p[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i])/4;
            #pragma omp parallel for
                for(j=1; j<Ndim*(Nlags+1)-i; j++) {
                    dcost[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i+j] /= Nsamp;
                    dcost[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i+j] += lambda*sign(p[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i+j]);
                    // dcost[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i+j] += 2*lambda*p[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i+j];
                    // dcost[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i+j] += lambda*sign(p[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i+j])/sqrt(fabs(p[1+(i+1)*Ndim*(Nlags+1)-i*(i+1)/2+i+j]))/2;
                }
        }
    }
    if(rank > 0 && rank <= Ndim*(Nlags+1)/2) {
        for(i=0; i<rank; i++) {
            dcost[1+Ndim*(Nlags+1)+rank*Ndim*(Nlags+1)+i] /= Nsamp;
            dcost[1+Ndim*(Nlags+1)+rank*Ndim*(Nlags+1)+i] += lambda*p[1+(rank+1)*Ndim*(Nlags+1)+i];
            for(j=0; j<Ndim*(Nlags+1); j++) {
                dcost[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)+j] /= Nsamp;
                dcost[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)+j] += lambda*p[1+(i+1)*Ndim*(Nlags+1)+j];
            }
        }
    }

    free(h);
    free(L);
    if(rank > 0) {
        free(TMP);
        free(J);
        free(COV);
        if(rank <= Ndim*(Nlags+1)/2) {
            free(U);
            free(UVT);
            free(VT);
            free(VV);
        }
    }

    return;

}

double sign(double x) {
    return (double)((x > 0) - (x < 0));
}



void gradCheck(double *analytic, double *empiric, double step, double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, long int *slice) {

    int size;
    long int i;

    if(rank > Ndim*(Nlags+1)/2) {
        size = 1+Ndim*(Nlags+1)+Ndim*(Nlags+1)*(Ndim*(Nlags+1)+1)/2;
    }
    else if(rank > 0) {
        size = 1+Ndim*(Nlags+1)+rank*(Ndim*(Nlags+1)+1);
    }
    else {
        size = 1+Ndim*(Nlags+1);
    }

    double *p_up = (double*)malloc(size * sizeof *p_up);
    double *p_down = (double*)malloc(size * sizeof *p_down);

    for(i=0; i<size; i++) {
        p_up[i] = p_down[i] = p[i];
    }

    p_up[0] = p[0]+step;
    p_down[0] = p[0]-step;

    empiric[0] = (Obj(p_up, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice) - Obj(p_down, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice))/(2.0*step);

    for(i=1; i<size; i++) {
        p_up[i-1] = p[i-1];
        p_down[i-1] = p[i-1];
        p_up[i] += step;
        p_down[i] -= step;
        empiric[i] = (Obj(p_up, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice) - Obj(p_down, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice))/(2.0*step);
    }

    dObj(analytic, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice);

    free(p_up);
    free(p_down);

    return;

}






/* Based on Numerical Recipes in C */
void frprmn(double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda,
                long int *trainInd, long int *cvInd, double ftol, int *iter, double *fret, double learning_rate, double momentum, int num_sched, double T, unsigned char do_sgd) {
    int n;
    double alpha = learning_rate;
    if(rank > Ndim*(Nlags+1)/2) {
        n = 1+Ndim*(Nlags+1)+Ndim*(Nlags+1)*(Ndim*(Nlags+1)+1)/2;
    }
    else if(rank > 0) {
        n = 1+Ndim*(Nlags+1)+rank*(Ndim*(Nlags+1)+1);
    }
    else {
        n = 1+Ndim*(Nlags+1);
    }

    int tally = 0;
    int i, j, its;
    double gg, gam, fp, dgg, cvfp, fbest, fepoch;
    long int stochInd[2];
    int Ntemp;
    int bad_start = 1;

    double *g = (double*)malloc(n * sizeof *g);
    double *h = (double*)malloc(n * sizeof *h);
    double *xi = (double*)malloc(n * sizeof *xi);
    double *ptrain = (double*)malloc(n * sizeof *ptrain);

    for(j=0; j<n; j++) {
        ptrain[j] = p[j];
    }

    fp = Obj(ptrain, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, trainInd);
    dObj(xi, ptrain, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, trainInd);
    cvfp = fbest = Obj(ptrain, rank, stim, Ndim, resp, Nsamples, Nlags, 0.0, cvInd);

    if(rank == 0 || rank > Ndim*(Nlags+1)/2) {
        for(j=0; j<n; j++) {
            g[j] = -xi[j];
            xi[j] = h[j] = g[j];
        }
    }
    else {
        fepoch = cvfp;
        for(j=0; j<n; j++) {
            g[j] = 0; //-learning_rate*xi[j];
            h[j] = p[j];
        }
        //h[0] = fp;
    }

    for(its=1; its<=ITMAX; its++) {

        *iter=its;

        if((rank > 0 && rank <= Ndim*(Nlags+1)/2) || do_sgd) {
            printf("iteration: %d;  alpha = %f;  fval = %lf;  cvfval = %lf;  fbest = %lf\n", its, alpha, fp, cvfp, fbest);

            if(trainInd[0] < trainInd[1]) {
                Ntemp = trainInd[1]-trainInd[0];
                stochInd[0] = trainInd[0]+(int)(1.0*rand()/RAND_MAX*(Ntemp-1));
            }
            else if(trainInd[0] > trainInd[1]) {
                Ntemp = Nsamples-Nlags-trainInd[0]+trainInd[1];
                stochInd[0] = (trainInd[0]+(int)(1.0*rand()/RAND_MAX*(Ntemp-1)))%(Nsamples-Nlags);
            }
            else {
                Ntemp = Nsamples-Nlags;
                stochInd[0] = trainInd[0]+(int)(1.0*rand()/RAND_MAX*(Ntemp-1));
            }
            stochInd[1] = (stochInd[0]+(int)(RAND_RATIO*Ntemp)-1);
            if(stochInd[1] > trainInd[1] && trainInd[1] > trainInd[0])
                stochInd[1] = trainInd[1];
            else if(stochInd[1] > Nsamples-Nlags && trainInd[1] < trainInd[0])
                    stochInd[1] = Nsamples-Nlags;

            fp = Obj(ptrain, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, stochInd);           
            dObj(xi, ptrain, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, stochInd);

            for(j=0; j<n; j++) {
                g[j] = momentum*g[j]-alpha*xi[j];
                ptrain[j] += g[j];
            }

            for(j=0; j<n; j++)
                ptrain[j] -= alpha*xi[j];

            cvfp = Obj(ptrain, rank, stim, Ndim, resp, Nsamples, Nlags, 0.0, cvInd);
            if(cvfp < fbest) {
                memcpy(h, ptrain, n * sizeof(double));
                memcpy(p, ptrain, n * sizeof(double));
                fepoch = fbest = cvfp;
            }
            else if(cvfp < fepoch) {
                memcpy(h, ptrain, n * sizeof(double));
                fepoch = cvfp;
            }
            else if(exp(-(cvfp-fepoch)/T) > 1.0*rand()/RAND_MAX) {
                memcpy(h, ptrain, n * sizeof(double));
                fepoch = cvfp;
            }
            else {
                memcpy(ptrain, h, n * sizeof(double));
            }

            tally++;

            if(tally++ >= num_sched) {
                alpha = learning_rate/(1+5.0*its/ITMAX);
                tally = 0;
            }
        }
        else {
            printf("iteration: %d;  fval = %f;  cvfval = %f;  fbest = %f;\n", its, fp, cvfp, fbest);

            dlinmin(xi,ptrain,rank,stim,Ndim,resp,Nsamples,Nlags,lambda,n,fret,trainInd);

            cvfp = Obj(ptrain, rank, stim, Ndim, resp, Nsamples, Nlags, 0.0, cvInd);
            if(cvfp < fbest) {
                fbest = cvfp;
                for(j=0; j<n; j++)
                    p[j] = ptrain[j];
                if(rank > Ndim*(Nlags+1)/2) {
                    for(i=0; i<Ndim*(Nlags+1); i++) {
                        for(j=0; j<Ndim*(Nlags+1)-i; j++) {
                            p[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i+j] = ptrain[1+Ndim*(Nlags+1)+i*Ndim*(Nlags+1)-i*(i+1)/2+i+j];
                        }
                    }
                }
                tally = 0;
            }
            else {
                tally++;
            }

            if((TALLY_MAX != 0 && tally > TALLY_MAX) && (rank == 0 || rank > Ndim*(Nlags+1)/2)) {
                printf("no better solution in %d tries\n", TALLY_MAX);
                free(g);
                free(h);
                free(xi);
                free(ptrain);
                return;
            }

            if(2.0*fabs(*fret-fp) <= ftol*(fabs(*fret)+fabs(fp)+EPS) && (rank == 0 || rank > Ndim*(Nlags+1)/2)) {
                printf("converged\n");
                free(h);
                free(g);
                free(xi);
                free(ptrain);
                return;
            }

            fp = Obj(ptrain, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, trainInd);
            dObj(xi, ptrain, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, trainInd);

            dgg = gg = 0.0;
            for(j=0; j<n; j++) {
                gg += g[j]*g[j];
                dgg += (xi[j]+g[j])*xi[j];
            }
            if(gg == 0.0) {
                free(g);
                free(h);
                free(xi);
                free(ptrain);
                return;
            }
            gam = dgg/gg;
            for(j=0; j<n; j++) {
                g[j] = -xi[j];
                xi[j] = h[j] = g[j]+gam*h[j];
            }
        }
    }
    printf("Too many iterations in frprmn\n");

    free(h);
    free(g);
    free(xi);
    free(ptrain);

    return;
}

void dlinmin(double *xi, double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, int n, double *fret, long int *slice) {

    int j;
    double xx,xmin,fx,fb,fa,bx,ax;

    ax=0.0;
    xx=0.2;//1.0;

    mnbrak(xi,p,rank,stim,Ndim,resp,Nsamples,Nlags,lambda,n,&ax,&xx,&bx,&fa,&fx,&fb,slice);
    *fret=dbrent(xi,p,rank,stim,Ndim,resp,Nsamples,Nlags,lambda,n,ax,xx,bx,TOL,&xmin,slice);
    #pragma omp for
        for(j=0; j<n; j++) {
            xi[j] *= xmin;
            p[j] += xi[j];
        }
}

void mnbrak(double *xi, double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags,
                double lambda, int n, double *ax, double *bx, double *cx, double *fa, double *fb, double *fc, long int *slice) {

    double ulim, u, r, q, fu, dum;

    *fa = f1dim(*ax, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);
    *fb = f1dim(*bx, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);

    if(*fb > *fa) {
        SHFT(dum,*ax,*bx,dum)
        SHFT(dum,*fb,*fa,dum)
    }

    *cx = *bx+GOLD*(*bx-*ax);
    *fc = f1dim(*cx, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);

    while(*fb > *fc) {
        r = (*bx-*ax)*(*fb-*fc);
	q = (*bx-*cx)*(*fb-*fa);
        u = (*bx)-((*bx-*cx)*q-(*bx-*ax)*r)/(2.0*SIGN(MAX(fabs(q-r),TINY),q-r));
        ulim = (*bx)+GLIMIT*(*cx-*bx);
        if((*bx-u)*(u-*cx) > 0.0) {
            fu = f1dim(u, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);
            if(fu < *fc) {
                *ax = (*bx);
                *bx = u;
                *fa = (*fb);
                *fb = fu;
                return;
            }
            else if(fu > *fb) {
                *cx = u;
                *fc = fu;
                return;
            }
            u = (*cx)+GOLD*(*cx-*bx);
            fu = f1dim(u, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);
        }
        else if((*cx-u)*(u-ulim) > 0.0) {
            fu = f1dim(u, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);
            if(fu < *fc) {
                SHFT(*bx,*cx,u,*cx+GOLD*(*cx-*bx))
                SHFT(*fb,*fc,fu,f1dim(u, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice))
            }
        }
        else if((u-ulim)*(ulim-*cx) >= 0.0) {
            u = ulim;
            fu = f1dim(u, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);
        }
        else {
            u = (*cx)+GOLD*(*cx-*bx);
            fu = f1dim(u, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);
        }
        SHFT(*ax,*bx,*cx,u)
        SHFT(*fa,*fb,*fc,fu)
    }
}

double dbrent(double *xi, double *p, int rank, double *stim, int Ndim, double *resp, long int Nsamples, int Nlags,
                double lambda, int n, double ax, double bx, double cx, double tol, double *xmin, long int *slice) {

    int iter,ok1,ok2;
    double a,b,d,d1,d2,du,dv,dw,dx,e=0.0;
    double fu,fv,fw,fx,olde,tol1,tol2,u,u1,u2,v,w,x,xm;

    a = (ax < cx ? ax : cx);
    b = (ax > cx ? ax : cx);
    x = w = v = bx;
    fw = fv = fx = f1dim(x, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);
    dw = dv = dx = df1dim(x, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);

    for(iter=1; iter<=ITMAX; iter++) {
        xm = 0.5*(a+b);
        tol1 = tol*fabs(x)+ZEPS;
        tol2 = 2.0*tol1;
        if(fabs(x-xm) <= (tol2-0.5*(b-a))) {
            *xmin=x;
            return fx;
        }
        if(fabs(e) > tol1) {
            d1 = 2.0*(b-a);
            d2 = d1;
            if (dw != dx)
                d1 = (w-x)*dx/(dx-dw);
            if (dv != dx)
                d2 = (v-x)*dx/(dx-dv);
            u1 = x+d1;
            u2 = x+d2;
            ok1 = (a-u1)*(u1-b) > 0.0 && dx*d1 <= 0.0;
            ok2 = (a-u2)*(u2-b) > 0.0 && dx*d2 <= 0.0;
            olde= e ;
            e = d;
            if (ok1 || ok2) {
                if (ok1 && ok2)
                    d = (fabs(d1) < fabs(d2) ? d1 : d2);
                else if (ok1)
                    d = d1;
                else
                    d = d2;
                if(fabs(d) <= fabs(0.5*olde)) {
                    u = x+d;
                    if(u-a < tol2 || b-u < tol2)
                        d = SIGN(tol1,xm-x);
                }
                else {
                    d = 0.5*(e=(dx >= 0.0 ? a-x : b-x));
                }
            }
            else {
                d = 0.5*(e=(dx >= 0.0 ? a-x : b-x));
            }
        }
        else {
            d = 0.5*(e=(dx >= 0.0 ? a-x : b-x));
        }
        if (fabs(d) >= tol1) {
            u = x+d;
            fu = f1dim(u, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);
        }
        else {
            u = x+SIGN(tol1,d);
            fu = f1dim(u, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);
            if(fu > fx) {
                *xmin = x;
                return fx;
            }
        }
        du = df1dim(u, xi, p, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, n, slice);
        if(fu <= fx) {
            if (u >= x)
                a=x;
            else
                b=x;
            MOV3(v,fv,dv,w,fw,dw)
            MOV3(w,fw,dw,x,fx,dx)
            MOV3(x,fx,dx,u,fu,du)
        }
        else {
            if (u < x)
                a=u;
            else
                b=u;
            if(fu <= fw || w == x) {
                MOV3(v,fv,dv,w,fw,dw)
                MOV3(w,fw,dw,u,fu,du)
            }
            else if(fu < fv || v == x || v == w) {
                MOV3(v,fv,dv,u,fu,du)
            }
        }
    }

    printf("Too many iterations in routine dbrent\n");

    return 0.0; //Never get here.
}


double f1dim(double x, double *xi, double *p, int rank, double *stim,
                int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, int n, long int *slice) {

    int j;
    double f, *xt;

    xt = (double*)malloc(n * sizeof *xt);

    for(j=0; j<n; j++)
        xt[j] = p[j]+x*xi[j];

    f = Obj(xt, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice);

    free(xt);

    return f;
}


double df1dim(double x, double *xi, double *p, int rank, double *stim,
                int Ndim, double *resp, long int Nsamples, int Nlags, double lambda, int n, long int *slice) {

    int j;
    double df1=0.0;
    double *xt,*df;

    xt=(double*)malloc(n * sizeof *xt);
    df=(double*)malloc(n * sizeof *df);

    for(j=0; j<n; j++)
        xt[j] = p[j]+x*xi[j];

    dObj(df, xt, rank, stim, Ndim, resp, Nsamples, Nlags, lambda, slice);

    for(j=0; j<n; j++)
        df1 += df[j]*xi[j];

    free(df);
    free(xt);

    return df1;
}
