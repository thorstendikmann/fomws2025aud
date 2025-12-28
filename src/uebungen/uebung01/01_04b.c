#include<stdio.h>
#include<math.h>
float max(float x, float y){
    return x>y?x:y;
}
int main(void){
    float x;
    float y;
    float z;
    printf("Input 3 Floats: x y z \n");
    int scan_return = scanf("%f %f %f", &x, &y, &z);
    if(scan_return != 3){
        printf("Those weren't 3 Foats! Bye.\n");
        return 1;
    }
    printf("Input x=%f, y=%f, z=%f\n", x,y,z);
    printf("fmax: %f\n",fmax(x,fmax(y,z)));
    printf("Max: %f\n", max(x,max(z,y)));
    return 0;
}