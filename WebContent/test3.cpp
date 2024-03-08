#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>

using namespace std;

class Fraction {
private:
    int num, den;
public:

    Fraction():num(1),den(1){}
    Fraction(int n):num(n),den(1){}
    Fraction(int n, int d):num(n),den(d){
        if(d<0){
            den=-d;
            num=-n;
        }
    };

    //计算公约数
    int gcd(int a,int b) const {
        if(b==0){
            return a;
        }
        return gcd(b,a%b);
    }


    //通分
    Fraction to_common(const Fraction &other) const {
        int common_Den =other.den*den / gcd(other.den,den);

        int new_num = num*(common_Den/den);
        return Fraction(new_num,common_Den);
    }
    //约分
    Fraction reduce(const Fraction &other) const {
        int div = gcd(other.num,other.den);
        int new_num=other.num/div;
        int new_den=other.den/div;
        return Fraction(new_num,new_den);
    }



    //重载函数
    Fraction operator + (const Fraction &f) const {
        int new_num=num*f.den+den*f.num;
        int new_den=den*f.den;
        Fraction newone=reduce(Fraction(new_num,new_den));
        return newone;
    }

    Fraction operator - (const Fraction &f) const {
        int new_num=num*f.den-den*f.num;
        int new_den=den*f.den;
        Fraction newone=reduce(Fraction(new_num,new_den));
        return newone;
    }
    Fraction operator * (const Fraction &f) const {
        int new_num=num*f.num;//分子相乘
        int new_den=den*f.den;//分母相乘
        Fraction newone=reduce(Fraction(new_num,new_den));
        return newone;
    }
    Fraction operator / (const Fraction &f) const {
        int new_num=num*f.den;//分子相乘
        int new_den=den*f.num;//分母相乘
        Fraction newone=reduce(Fraction(new_num,new_den));
        return newone;
    }
    bool operator == (const Fraction &f) const {
        Fraction com_n1 = to_common(f);
        Fraction com_n2 = f.to_common(*this);
        return com_n1.num == com_n2.num;
    }
    bool operator != (const Fraction &f) const {
        Fraction com_n1 = to_common(f);
        Fraction com_n2 = f.to_common(*this);
        return com_n1.num != com_n2.num;
    }

    bool operator <(const Fraction &f) const {
        Fraction com_n1 = to_common(f);
        Fraction com_n2 = f.to_common(*this);
        return com_n1.num < com_n2.num;
    }

    bool operator >(const Fraction &f) const {
        Fraction com_n1 = to_common(f);
        Fraction com_n2 = f.to_common(*this);
        return com_n1.num > com_n2.num;
    }

    bool operator <=(const Fraction &f) const {
        Fraction com_n1 = to_common(f);
        Fraction com_n2 = f.to_common(*this);
        return com_n1.num <= com_n2.num;
    }

    bool operator >=(const Fraction &f) const {
        Fraction com_n1 = to_common(f);
        Fraction com_n2 = f.to_common(*this);
        return com_n1.num >= com_n2.num;
    }

    friend ostream& operator << (ostream &out, const Fraction &f);
    friend istream& operator >> (istream &in, Fraction &f);
};

istream& operator >> (istream &in, Fraction &f){
    int n,d;
    in >> n >> d;
    f = Fraction(n,d);
    return in;
}

ostream& operator << (ostream &out, const Fraction &f){
    out << f.num << "/" << f.den;
    return out;
}

int main(int argc, char *argv[]) {
    
    Fraction a(1),b(1,3),c(-3,9),d(2,-6);
    
    cin>>a>>b;
    
    cout << "a= " << a << ", b = " << b << endl;
    
    cout << a << " + " << b << " = " << a + b << endl;
    cout << a << " - " << b << " = " << a - b << endl;
    cout << a << " * " << b << " = " << a * b << endl;
    cout << a << " / " << b << " = " << a / b << endl;
    
    cout << "a == b is " << boolalpha << (a == b) << endl;
    cout << "a != b is " << boolalpha << (a != b) << endl;
    cout << "a <= b is " << boolalpha << (a <= b) << endl;
    cout << "a >= b is " << boolalpha << (a >= b) << endl;
    cout << "a < b is " << boolalpha << (a < b) << endl;
    cout << "a > b is " << boolalpha << (a > b) << endl;
    cout << "c == d is " << boolalpha << (c == d) << endl;
    
    return 1;
}
