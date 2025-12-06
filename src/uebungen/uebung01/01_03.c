#include <stdio.h>
int main(void)
{
int days = 1331;
int years = days / 365;
days = days % 365;
printf("%i years, %i days left to calculate\n", years, days);
int months = days / 30;
days = days % 30;
printf("%i months, %i days left\n", months, days);
printf("%i years, %i months, %i days\n", years, months, days);
return 0;
}