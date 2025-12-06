1.
a) n
b) log(n)
c) 2^n
d) n ( Faktor fällt weg)

2.
a) n^2
b) n^4
c) n^2



int f5(int n)
{
int x = 1;              // 1
int k = f1(n);          // n^2
while (k > 0)           // n 
{                       
x += f2(n);             // log(n)
k--;                    // 1
}                       // n * (1 + log(n))
return x;               // 1
}