#include <cstdio>
#include <vector>
#include <algorithm>
using namespace std;

const int MAXN = 50;

struct Node {
    int data;
    int height;
    int l, r;
} nodes[MAXN];

int nodeCount = 0;

int newNode(int data) {
    nodes[nodeCount].data = data;
    nodes[nodeCount].height = 1;
    nodes[nodeCount].l = nodes[nodeCount].r = -1;
    return nodeCount++;
}

int getHeight(int root) {
    if (root == -1) {
        return 0;
    } else {
        return nodes[root].height;
    }
}

void updateHeight(int root) {
    nodes[root].height = max(getHeight(nodes[root].l), getHeight(nodes[root].r)) + 1;
}

int getBalanceFactor(int root) {
    return getHeight(nodes[root].l) - getHeight(nodes[root].r);
}

int L(int root) {
    int temp = nodes[root].r;
    nodes[root].r = nodes[temp].l;
    nodes[temp].l = root;
    updateHeight(root);
    updateHeight(temp);
    return temp;
}

int R(int root) {
    int temp = nodes[root].l;
    nodes[root].l = nodes[temp].r;
    nodes[temp].r = root;
    updateHeight(root);
    updateHeight(temp);
    return temp;
}

int insert(int root, int data) {
    if (root == -1) {
        return newNode(data);
    }
    if (data < nodes[root].data) {
        nodes[root].l = insert(nodes[root].l, data);
        updateHeight(root);
        if (getBalanceFactor(root) == 2) {
            if (getBalanceFactor(nodes[root].l) == 1) {
                root = R(root);
            } else if (getBalanceFactor(nodes[root].l) == -1) {
                nodes[root].l = L(nodes[root].l);
                root = R(root);
            }
        }
    } else {
        nodes[root].r = insert(nodes[root].r, data);
        updateHeight(root);
        if (getBalanceFactor(root) == -2) {
            if (getBalanceFactor(nodes[root].r) == -1) {
                root = L(root);
            } else if (getBalanceFactor(nodes[root].r) == 1) {
                nodes[root].r = R(nodes[root].r);
                root = L(root);
            }
        }
    }
    return root;
}

vector<int> pre;

void preOrder(int root) {
    if (root == -1) {
        return;
    }
    pre.push_back(nodes[root].data);
    preOrder(nodes[root].l);
    preOrder(nodes[root].r);
}

int main() {
    int n, data, root = -1;
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &data);
        root = insert(root, data);
    }
    preOrder(root);
    for (int i = 0; i < (int)pre.size(); i++) {
        printf("%d", pre[i]);
        if (i < (int)pre.size() - 1) {
            printf(" ");
        }
    }
    return 0;
}