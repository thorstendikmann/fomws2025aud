#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
bool isPrime(int num){
    for(int i = num - 1; i > 1;i--){
        if(num % i == 0){
            return false;
        }
    }
    return true;
}
void findPrimesBasic(int max){
    for (int i = 2; i <= max; i++){
        if(isPrime(i)){
           //no output, we're happy just computing it
        }
    }
    
}
void findPrimesSieve(int max){
    bool crossed_out[max + 1];
    //init array
    for (int i = 2; i <= max; i++) {
        crossed_out[i] = false;
    }

    //iterate to sqrt(MAX)
    for(int i =2; i < sqrt(max); i++){
        //only inspect primes
        if(!crossed_out[i]){
            //cross out multiples of chosen prime
            for(int j = i*i; j<max; j = j+i){
                crossed_out[j] = true;
            }
        }
    }
}



int main(void){

    int MAX = 5000;
    struct timespec begin, after_basic, after_sieve;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin);
    findPrimesBasic(MAX);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &after_basic);

     printf ("Basic Algo took %f seconds\n",
            (after_basic.tv_nsec - begin.tv_nsec) / 1000000000.0 +
            (after_basic.tv_sec  - begin.tv_sec));

    findPrimesSieve(MAX);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &after_sieve);
    printf ("Sieve Algo took %f seconds\n",
            (after_sieve.tv_nsec - after_basic.tv_nsec) / 1000000000.0 +
            (after_sieve.tv_sec  - after_basic.tv_sec));
            
    return 0;
}