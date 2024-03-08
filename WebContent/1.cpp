// 本题主要考察-构造函数的定义，根据后缀，和程序输出完成分数类的构造函数定义。
// 输入：
// 无
// 输出：
// 1/1
// 2/1
// 1/3

#include <iostream>

using namespace std;
class Fraction{
    private:
        int numerator, denominator;
    public:
        Fraction(){ 
            numerator=1;
            denominator=1;
        }
        Fraction(int n){
            numerator=n;
            denominator=1;
        }
        Fraction(int n, int d){
            numerator=n;
            denominator=d; 
        }
        void show(){
            cout << numerator << "/" << denominator;
        }
};
	


//StudybarCommentBegin
int main()
{
	Fraction f1,f2(2),f3(1,3);
	f1.show(); cout<<endl;
	f2.show(); cout<<endl;
	f3.show(); cout<<endl;
}
//StudybarCommentEnd

