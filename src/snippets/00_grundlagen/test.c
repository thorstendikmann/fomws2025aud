#include <stdio.h>
int main(void){
    printf("Hello World!\n");
    char ch = "h";
    int i = 42;
    printf("String: %s @ %p\n", ch);
    printf("Int: %i @ %p\n", i, (void*) i*);
    printf("Float: %f\n", 33.33);
}