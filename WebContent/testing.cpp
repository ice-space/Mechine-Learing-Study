// 本题主要考察-构造函数的定义和操作符重载等
// 根据后缀，和程序输出完成分数类和fswap函数的定义，
// 包括：
// 构造函数。
// Show函数。
// 操作符重载。
// fswap函数。
// 输入：
// 2
// 3

// 输出：

// 1/3
// 2/1
// 1/6

#include <iostream>
#include <cmath>
#include <string>

using namespace std;


class Fraction{
    private:
        int num,den;
    public:
        Fraction(int x):num(x),den(1){}
        Fraction(int x,int y):num(x),den(y){}

        void Show() const {
            cout<<num<<"/"<<den;
        }


        //计算公约数
        int gcd(int a,int b) const {
            if(b==0){
                return a;
            }
            return gcd(b,a%b);
        }

        //约分
        Fraction reduce(const Fraction &other) const {
            int div = gcd(other.num,other.den);
            int new_num=other.num/div;
            int new_den=other.den/div;
            return Fraction(new_num,new_den);
        }

        Fraction operator - (const Fraction &f) const {
            int new_num=num*f.den-den*f.num;
            int new_den=den*f.den;
            Fraction newone=reduce(Fraction(new_num,new_den));
            return newone;
        }
};

void fswap(Fraction &a,Fraction &b){
    Fraction temp=a; 
    a=b; 
    b=temp;
}


//StudybarCommentBegin
int main(int argc, char *argv[]) {
	int x,y;
	cin>>x>>y;
    Fraction a(1,x),b(1,y),c(2);
    fswap(a,b);
    a.Show(); cout<<endl;
    c.Show(); cout<<endl;
    c=b-a;
    c.Show();    
    return 1;
}
//StudybarCommentEnd