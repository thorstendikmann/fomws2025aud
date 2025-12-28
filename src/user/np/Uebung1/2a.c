#include <stdio.h>
typedef struct
{
    float breite;
    float laenge;
} rechteck;
void calcAndPrintUmfang(rechteck *r)
{
    printf("Rechteck Laenge %f", r->laenge);

    float umfang = r->breite *2 + r->laenge *2;

    printf("Rechteck Umfang %f", umfang);
    
    //printf("Umfang: %d", r->laenge *2 + *r->breite *2);
}
void calcAndPrintFlaeche(rechteck *r)
{

}
int main()
{
    rechteck reck;
    reck.breite = 10.0;
    reck.laenge = 5.0;
    printf("Rechteck: Länge: %f, Breite: %f\n",
           reck.laenge, reck.breite);
    calcAndPrintUmfang(&reck);
    calcAndPrintFlaeche(&reck);
    return 0;
}