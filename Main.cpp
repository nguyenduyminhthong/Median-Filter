#include <bits/stdc++.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "MedianFilter.h"

using namespace cv;
using namespace std;

const int EPOCHS = 1;

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);
	Mat input = imread("Input.jpg", IMREAD_GRAYSCALE);
	Mat output = input.clone();

	float time = 0;
	time += MedianFilter(input, output);
	for (int i = 1; i < EPOCHS; ++i) time += MedianFilter(output, output);

	cout << "Time: " << time << "s\n";

	imshow("Input", input);
	imshow("Output", output);
	imwrite("Output.jpg", output);
	while (waitKey(1) != 'q') {
	}
}
