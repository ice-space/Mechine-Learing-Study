// 本题主要考察-构造函数的定义和操作符重载、友元函数等

// 根据后缀和程序样例输出，完成时钟类和相关函数的定义，

// 输入：

// 1 2 3

// 输出：

// t0 is 12:01:02 AM (注意0点的12小时制为 12:XX:XX AM )
// t0是一天中的62秒
// t1 is 12:01:02 PM (12点的12小时制显示为 12:XX:XX PM)
// t2 is 00:01:02
// t3 is 23:45:02
// (++t3)++ is 11:45:03 PM
// t3 is 11:45:04 PM
// (++(++(++t2)))++ is 01:02:06
// t2的12小时制：01:02:07 AM
// t2是一天中的3727秒

// 提示，部分参考代码如下：

// #include <iostream>
// #include <iomanip>
// using namespace std;

// class Clock24 {
// private:
// int hour,minute,second;
// public:
// bool IS24;
// Clock24(int h=0,int m=0,int s=0,bool is24=true):IS24(is24),hour(h%24),minute(m%60),second(s%60) {}
// int getSecondsOfDay() const;

// Clock24& operator++();
// Clock24 operator++(int);

// friend istream& operator>>(istream & in,Clock24 &c);
// friend ostream& operator<<(ostream & out,const Clock24 &c);
// };

// ostream& operator<<(ostream & out,const Clock24 &c)
// {
// if (c.IS24)
// out<<setfill('0')<<setw(2)<<c.hour<<":"<<setw(2)<<c.minute<<":"<<setw(2)<<c.second;
// else
// out<<setfill('0')<<setw(2)<< (c.hour%12==0? 12:c.hour%12 )<<":"<<setw(2)<<c.minute<<":"<<setw(2)<<c.second << (c.hour>=12?" PM":" AM");
// return out;
// }

#include <iostream>
#include <iomanip>
using namespace std;

class Clock24 {
private:
int hour,minute,second;
public:
    bool IS24;
    Clock24(int h=0,int m=0,int s=0,bool is24=true):IS24(is24),hour(h%24),minute(m%60),second(s%60) {}
    
    int getSecondsOfDay() const{
        long out = hour*3600 + minute*60 + second; 
        return out;
    }

    Clock24& operator++(){
            second++;
            while (second >= 60) {
                second -= 60;
                minute++;
                if (minute >= 60) {
                    minute -= 60;
                    hour = (hour + 1) % 24;
                }
            }
        return *this;
    }




    Clock24 operator++(int){
        Clock24 temp(*this); // 创建一个副本
        ++(*this); // 调用前置++运算符
        return temp; // 返回副本
    }

friend istream& operator>>(istream & in,Clock24 &c);
friend ostream& operator<<(ostream & out,const Clock24 &c);
};



istream& operator>>(istream & in,Clock24 &c){
    in >> c.hour >> c.minute >> c.second;
    return in;
}


ostream& operator<<(ostream & out,const Clock24 &c)
{
if (c.IS24)
out<<setfill('0')<<setw(2)<<c.hour<<":"<<setw(2)<<c.minute<<":"<<setw(2)<<c.second;
else
out<<setfill('0')<<setw(2)<< (c.hour%12==0? 12:c.hour%12 )<<":"<<setw(2)<<c.minute<<":"<<setw(2)<<c.second << (c.hour>=12?" PM":" AM");
return out;
}







	
//StudybarCommentBegin
int main()
{
    const Clock24 t0(0,1,2,false); //第四个参数若为false,表示用12小时制显示时间。 
    cout<<"t0 is "<< t0 <<" (注意0点的12小时制为 12:XX:XX AM )";
    cout<<"\nt0是一天中的"<<t0.getSecondsOfDay()<<"秒";
    Clock24 t1(12,1,2,false), t2(t0), t3(23,45,2);       
	cout<<"\nt1 is "<<t1<<" (12点的12小时制显示为 12:XX:XX PM) ";
	t2.IS24=true; //用于设置是24小时制(true),还是12小时制(false)的显示。 
	cout<<"\nt2 is "<<t2<<"\nt3 is "<<t3<<endl;   
    t3.IS24=false;
    cout<<"(++t3)++ is "<<(++t3)++<<endl;
    cout<<"t3 is "<<t3<<endl;

    cin >> t2;
    cout<<"(++(++(++t2)))++ is "<<(++(++(++t2)))++<<endl;
    t2.IS24=false;
    cout<<"t2的12小时制："<<t2;
    cout<<"\nt2是一天中的"<<t2.getSecondsOfDay()<<"秒"<<endl;

    return 0;
}
//StudybarCommentEnd