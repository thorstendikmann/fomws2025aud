#include <stdio.h>
#include <stdbool.h>
bool isPrime(int num){
    for(int i = num - 1; i > 1;i--){
        if(num % i == 0){
            return false;
        }
    }
    return true;
}
int main(void){
    for (int i = 2; i <= 1000; i++)
    {
        if(isPrime(i)){
            printf("%i\n",i);
        }
    }
    
    return 0;
}