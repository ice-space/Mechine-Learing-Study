// 系数为复数的一元二次方程的解：
// 要求：1、输入三个系数的实部和虚部，输出两个解（可能相同，也可能不同）。
//             2、两个解不同时，需先输出虚部为正的解。
//             3、对虚部为负的解，输出形式因如样例1。
//                   对虚部为0的解，输出形式因如样例2.
// 提示：1、方程可以用求根公式求解。
//             2、需考虑当判别式为负实数时的情况。
// 样例1输入：
// 1 1
// 1 1
// 1 1
// 输出：
// (-0.5+0.866025i)
// (-0.5-0.866025i)
// 样例2输入：
// 1 0
// -1 -1
// 0 0
// 输出：
// (1+1i)
// (0+0i)

#include <iostream>
#include <string>
#include <cmath>

using namespace std;

class Cmycomplex
{
    private:
        double real, imag;
    public:
        // 构造函数
        Cmycomplex(double r,double i):real(r),imag(i){} 
        Cmycomplex():real(0),imag(0){}



        //获取函数
        double getx() const {
            return real;
        }
        double gety() const {
            return imag;        
        }
        void Show(){
            // cout <<setiosflags(ios::fixed);
            cout<<"(" << real <<(imag>=0?"+":"") << imag<<"i)" << endl;
        }
        
        // 开方
        Cmycomplex sqrt() const { // 开方运算符重载，非友元重载
            double sqrtPart = ::sqrt( pow(this->real, 2) + pow(this->imag, 2) );
            double newReal = ::sqrt((this->real + sqrtPart) / 2);
            double newImag = this->imag / ::sqrt( 2* (this->real + sqrtPart));
            return Cmycomplex(newReal, newImag);
        }

        //友元重载
        friend Cmycomplex operator*(int num, const Cmycomplex &other); // 整数乘以复数
        friend Cmycomplex operator*(const Cmycomplex &other, int num); // 复数乘以整数
        friend Cmycomplex operator*(const Cmycomplex &c1, const Cmycomplex &c2); // 复数乘以复数

        friend Cmycomplex operator/(int num, const Cmycomplex &other); // 整数除以复数
        friend Cmycomplex operator/(const Cmycomplex &other, int num); // 复数除以整数
        friend Cmycomplex operator/(const Cmycomplex &c1, const Cmycomplex &c2); // 复数除以复数

        friend Cmycomplex operator+(int num, const Cmycomplex &other); // 整数加上复数
        friend Cmycomplex operator+(const Cmycomplex &other, int num); // 复数加上整数
        friend Cmycomplex operator+(const Cmycomplex &c1, const Cmycomplex &c2); // 复数加上复数

        friend Cmycomplex operator-(int num, const Cmycomplex &other); // 整数减去复数
        friend Cmycomplex operator-(const Cmycomplex &other, int num); // 复数减去整数
        friend Cmycomplex operator-(const Cmycomplex &c1, const Cmycomplex &c2); // 复数减去复数

};
// 整数加上复数
Cmycomplex operator+(int num, const Cmycomplex &other) {
    double newReal = num + other.real;
    double newImag = other.imag;
    return Cmycomplex(newReal, newImag);
}

// 复数加上整数
Cmycomplex operator+(const Cmycomplex &other, int num) {
    double newReal = other.real + num;
    double newImag = other.imag;
    return Cmycomplex(newReal, newImag);
}

// 复数加上复数
Cmycomplex operator+(const Cmycomplex &c1, const Cmycomplex &c2) {
    double newReal = c1.real + c2.real;
    double newImag = c1.imag + c2.imag;
    return Cmycomplex(newReal, newImag);
}

// 整数减去复数
Cmycomplex operator-(int num, const Cmycomplex &other) {
    double newReal = num - other.real;
    double newImag = -other.imag;
    return Cmycomplex(newReal, newImag);
}

// 复数减去整数
Cmycomplex operator-(const Cmycomplex &other, int num) {
    double newReal = other.real - num;
    double newImag = other.imag;
    return Cmycomplex(newReal, newImag);
}

// 复数减去复数
Cmycomplex operator-(const Cmycomplex &c1, const Cmycomplex &c2) {
    double newReal = c1.real - c2.real;
    double newImag = c1.imag - c2.imag;
    return Cmycomplex(newReal, newImag);
}

// 整数乘以复数
Cmycomplex operator*(int num, const Cmycomplex &other) {
    double newReal = num * other.real;
    double newImag = num * other.imag;
    return Cmycomplex(newReal, newImag);
}

// 复数乘以整数
Cmycomplex operator*(const Cmycomplex &other, int num) {
    double newReal = num * other.real;
    double newImag = num * other.imag;
    return Cmycomplex(newReal, newImag);
}

// 复数乘以复数
Cmycomplex operator*(const Cmycomplex &c1, const Cmycomplex &c2) {
    double newReal = c1.real * c2.real - c1.imag * c2.imag;
    double newImag = c1.imag * c2.real + c1.real * c2.imag;
    return Cmycomplex(newReal, newImag);
}

// 整数除以复数
Cmycomplex operator/(int num, const Cmycomplex &other) {
    if (other.real == 0 && other.imag == 0) {
        cout << "Error: Division by zero" << endl;
        return Cmycomplex(0, 0);
    }
    double temp = other.real * other.real + other.imag * other.imag;
    double newReal = (num * other.real) / temp;
    double newImag = (-num * other.imag) / temp;
    return Cmycomplex(newReal, newImag);
}

// 复数除以整数
Cmycomplex operator/(const Cmycomplex &other, int num) {
    if (num == 0) {
        cout << "Error: Division by zero" << endl;
        return Cmycomplex(0, 0);
    }
    double newReal = other.real / num;
    double newImag = other.imag / num;
    return Cmycomplex(newReal, newImag);
}

// 复数除以复数
Cmycomplex operator/(const Cmycomplex &c1, const Cmycomplex &c2) {
    if (c2.real == 0 && c2.imag == 0) {
        cout << "Error: Division by zero" << endl;
        return Cmycomplex(0, 0);
    }
    double temp = c2.real * c2.real + c2.imag * c2.imag;
    double newReal = (c1.real * c2.real + c1.imag * c2.imag) / temp;
    double newImag = (c1.imag * c2.real - c1.real * c2.imag) / temp;
    return Cmycomplex(newReal, newImag);
}



//StudybarCommentBegin
int main()
{
	double x1, x2, x3, y1, y2, y3;
	
	cin >> x1 >> y1;
	cin >> x2 >> y2;
	cin >> x3 >> y3;

	Cmycomplex a(x1, y1), b(x2, y2), c(x3, y3), z, t1, t2;

	z=b*b-4*a*c;

	t1=((-1)*b+z.sqrt())/(2*a);     //z.sqrt()为求复数z的平方根，这里的2*a涉及操作符重载，且友元重载。
	t2=((-1)*b-z.sqrt())/2.0/a;   //这里涉及到除法的重载

	if(t1.gety()>t2.gety())   //gety()为得到虚部值
	{
		t1.Show();
		t2.Show();
	}
	else
	{
		t2.Show();
		t1.Show();
	}
	return 0;
}
//StudybarCommentEnd