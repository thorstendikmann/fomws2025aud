#include <stdio.h>
#include <stdbool.h>
#include <math.h>

int main(void)
{
    int MAX = 1000;
    // offset one to save braincells getting from by index
    bool crossed_out[MAX + 1];

    //init array
    for (int i = 2; i <= MAX; i++) {
        crossed_out[i] = false;
    }

    //iterate to sqrt(MAX)
    for(int i =2; i < sqrt(MAX); i++){
        //only inspect primes
        if(!crossed_out[i]){
            //cross out multiples of chosen prime
            for(int j = i*i; j<MAX; j = j+i){
                crossed_out[j] = true;
            }
        }
    }

    //print solution
    printf("Found the following primes:\n");
    for (int i = 2; i <= MAX; i++){
        if (!crossed_out[i]) {
            printf("%i\n", i);
        }
    }

    return 0;
}