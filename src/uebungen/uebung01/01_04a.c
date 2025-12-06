#include<stdio.h>
#include<math.h>
int main(void){
    int x;
    int y;
    printf("Input 2 Integers: x y\n");
    int scan_return = scanf("%i %i", &x, &y);
    if(scan_return != 2){
        printf("Those weren't 2 Integers! Bye.\n");
        return 1;
    }
    printf("Input x=%i, y=%i\n", x,y);
    printf("x+y=%i\n",x+y);
    printf("x*y=%i\n",x*y);
    printf("x^y=%f\n",pow(x,y));
    return 0;
}