#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include <iostream>

#include "matrix.h"

using namespace std;
using namespace YAMATH;

void pr(const MatrixGpu &inM)
{
    MatrixCpu m = inM;

    t_index len = 5;
    t_index tot = m.getX()*m.getY();

    cout << "[ ";

    if(tot > len*3)
    {
        for(t_index i = 0; i < len; ++i)
        {
            cout << " " << m.getData()[i];
        }

        cout << " ...";

        for(t_index i = (tot + len)/2; i < (tot + len)/2 + len; ++i)
        {
            cout << " " << m.getData()[i];
        }

        cout << " ...";

        for(t_index i = tot - len; i < tot; ++i)
        {
            cout << " " << m.getData()[i];
        }
    }
    else
    {
        for(t_index i = 0; i < tot; ++i)
        {
            cout << " " << m.getData()[i];
        }
    }

    cout << " ]";
}

#define pr1(m) cout << #m << ": " << (m).getX() << " x " << (m).getY() << endl
#define pr2(m) cout << #m << ": " << (m).getX() << " x " << (m).getY() << ": "; pr(m); cout << endl;

int main(void)
{
    MatrixGpu a,b,c,d;

    d.Reset(1000, 1000);
    a = b = c = d;

    a.RandNormal(0.0f, 1.0f);
    b = 11.0;
    c = 12.0;
    d = 13.0;

    b += a;

    d = c*a;
    c *= a;

    pr2(a);
    pr2(b);
    pr2(c);
    pr2(d);


//    cout << sysconf(_SC_PAGE_SIZE) << endl;

    return 0;
}
