#include <iostream>
#include <cmath>
#include <cstdio>
#include <string.h>
#include <string>
#include <algorithm>
#include <vector>
using namespace std;
/*平衡二叉树 AVL树  左子树与右子树高度之差叫平衡因子 绝对值不大于1
*/
//平衡二叉树的建立
const int M = 100;
int ks;
struct node{
    int data,lchild,rchild,height;
} arr[M];
int ind = 0;
int newnode(int x){
    arr[ind].data = x;
    arr[ind].height = 1;
    arr[ind].lchild = arr[ind].rchild = -1;
    return ind++;
}
//获得高度
int getheight(int root){
    if(root==-1){
        return 0;
    }
    return arr[root].height;
}
//计算平衡因子  使用arr[arr[root].lchild].height 可能没有右子树 所以还是用函数
int balancefactor(int root){
    return getheight(arr[root].lchild) - getheight(arr[root].rchild);
}
//更新高度
void updateheight(int root){
    arr[root].height = max(getheight(arr[root].lchild), getheight(arr[root].rchild)) + 1;
}
//查找和二叉搜索树一样
//插入前置准备 左旋  看书以根为起点进行旋转
int lxuan(int root){
    int temp = arr[root].rchild;
    arr[root].rchild = arr[temp].lchild;
    arr[temp].lchild = root;
    updateheight(root);
    updateheight(temp);
    return temp;
}
int rxuan(int root){
    int temp = arr[root].lchild;
    arr[root].lchild = arr[temp].rchild;
    arr[temp].rchild = root;
    updateheight(root);
    updateheight(temp);
    return temp;
}
//每次操作都要返回新的根
int insert(int root,int v){
    if(root==-1){
        int newroot = newnode(v);
        return newroot;
    }
    if(v<arr[root].data){
        arr[root].lchild = insert(arr[root].lchild, v);
        updateheight(root);
        if(balancefactor(root)==2){
            if(balancefactor(arr[root].lchild)==1){  //ll型
                root = rxuan(root);
            }else if(balancefactor(arr[root].lchild)==-1){
                // 谁旋下去了 以谁为起点
                //看书的图 lr型左旋以左孩子为起点 然后以根为起点
                arr[root].lchild = lxuan(arr[root].lchild); //通过赋值修改左孩子指针
                root = rxuan(root);
            }
        }
    }else{
        arr[root].rchild = insert(arr[root].rchild, v);
        updateheight(root);
        if(balancefactor(root)==-2){
            if(balancefactor(arr[root].rchild)==-1){
                root = lxuan(root);
            }else if(balancefactor(arr[root].rchild)==1){
                arr[root].rchild = rxuan(arr[root].rchild);
                root = lxuan(root);
            }
        }
    }
    return root;  //重点看别忘了
}
// void preorder(int root){
//     if(root==-1){
//         return;
//     }
//     printf("%d", arr[root].data);
//     ks--; //控制空格输出
//     if(ks>0){
//         printf(" ");
//     }
//     preorder(arr[root].lchild);
//     preorder(arr[root].rchild);
// }
vector<int> pre;

void preOrder(int root) {
    if (root == -1) {
        return;
    }
    pre.push_back(arr[root].data);
    preOrder(arr[root].lchild);
    preOrder(arr[root].rchild);
}

int main(){
    int n,data,root=-1;
    cin >> n;
    // ks = n;
    for (int i = 0; i < n; i++){
        scanf("%d", &data);
        root = insert(root, data);
    }
    // preorder(root);
    preOrder(root);
    for (int i = 0; i < (int)pre.size(); i++) {
        printf("%d", pre[i]);
        if (i < (int)pre.size() - 1) {
            printf(" ");
        }
    }
}
