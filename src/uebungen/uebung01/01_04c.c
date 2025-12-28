#include <stdio.h>
#include <string.h>
typedef struct
{
    char name[100];
    char address[100];
    int plz;
    char city[100];
} user;

void printUser(user *user)
{
    printf("Name: %s\n", user->name);
    printf("Address: %s\n", user->address);
    printf("PLZ: %i\n", user->plz);
    printf("City: %s\n", user->city);
}
int main(void)
{
    user input_user;

    printf("Input Name\n");
    fgets(input_user.name, 100, stdin);
    //replace newline with string terminator
    //strcspn returns length of string segment without \n, so length as index (since it's 0 based) returns the position of our trailing \n
    input_user.name[strcspn(input_user.name, "\n")] = 0;
    
    //yes, yes, we could move that to a subfunction since it's a 3x repetition
    printf("Input Address\n");
    fgets(input_user.address, 100, stdin);
    input_user.address[strcspn(input_user.address, "\n")] = 0;
    
    
    printf("Input PLZ\n");
    scanf("%i", &(input_user.plz));

    //stop the trailing \n scanf doesn't consume
    int c;
    while ((c = getchar()) != '\n' && c != EOF) { }

    printf("Input City\n");
    fgets(input_user.city, 100, stdin);
    input_user.city[strcspn(input_user.city, "\n")] = 0;
    printf("Inputs done\n");
    printUser(&input_user);
    return 0;
}