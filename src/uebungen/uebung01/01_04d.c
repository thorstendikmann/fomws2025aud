#include<stdio.h>
#include<math.h>
int main(void){
    float x;
    float y;
    float z;
    printf("Input 3 Floats: x y z\n");
    int scan_return = scanf("%f %f %f", &x, &y, &z);
    if(scan_return != 3){
        printf("Those weren't 3 Floats! Bye.\n");
        return 1;
    }
    if(x+y > z && x+z > y && y+z > x){
        printf("That's a valid triangle\n");
        
    } else{
        printf("That's not a valid triangle\n");
    }
    return 0;
}