#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

int main() {
    double a_r, b_r, c_r;
    double a_y, b_y, c_y;

    cin >> a_r >> a_y;
    cin >> b_r >> b_y;
    cin >> c_r >> c_y;

    double delta_r = pow(b_r, 2) - 4 * a_r * c_r;
    double delta_y = 2 * b_r * b_y - 2 * a_r * c_y - 2 * a_y * b_r;

    if (delta_r < 0) {
        cout << fixed << setprecision(2) << "(" << -b_r / (2 * a_r) << "+" << sqrt(-delta_r) / (2 * a_r) << "i)" << endl;
        cout << fixed << setprecision(2) << "(" << -b_r / (2 * a_r) << "-" << sqrt(-delta_r) / (2 * a_r) << "i)" << endl;
    } else if (delta_r > 0) {
        double x1_r = (-b_r + sqrt(delta_r)) / (2 * a_r);
        double x2_r = (-b_r - sqrt(delta_r)) / (2 * a_r);

        // Output the solutions in the desired format
        if (x1_r == x2_r) {
            cout << fixed << setprecision(2) << "(" << x1_r << "+0.00i)" << endl;
            cout << fixed << setprecision(2) << "(" << x2_r << "+0.00i)" << endl;
        } else if (x1_r > x2_r) {
            cout << fixed << setprecision(2) << "(" << x1_r << "+0.00i)" << endl;
            cout << fixed << setprecision(2) << "(" <<x2_r<< "+0.00i)" << endl;
        } else {
            cout << fixed << setprecision(2) << "(" << x2_r << "+0.00i)" << endl;
            cout << fixed << setprecision(2) << "(" << x1_r << "+0.00i)" << endl;
        }
    } else {
        double x_r = -b_r / (2 * a_r);
        double x_y = -b_y / (2 * a_r);

        // Output the solution in the desired format
        if (x_y < 0) {
            cout << fixed << setprecision(2) << "(" << x_r << "-" << -x_y << "i)" << endl;
            cout << fixed << setprecision(2) << "(" << x_r << "+" << -x_y << "i)" << endl;
        } else if (x_y == 0) {
            cout << fixed << setprecision(2) << "(" << x_r << "+0.00i)" << endl;
            cout << fixed << setprecision(2) << "(" << x_r << "+0.00i)" << endl;
        } else {
            cout << fixed << setprecision(2) << "(" << x_r << "+" << x_y << "i)" << endl;
            cout << fixed << setprecision(2) << "(" << x_r << "-" << x_y << "i)" << endl;
        }
    }

    return 0;
}
