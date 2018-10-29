#include<iostream>
#include<opencv2/opencv.hpp>
#include<thread>
using namespace std;

void work(std::string address, std::string window) {
    cv::VideoCapture cap(address);
    if (!cap.isOpened()) {
        std::cout << "Cannot open camera" << std::endl;
        return;
    }
    cv::Mat frame;
    while (char(cv::waitKey(1)) != 'q' && cap.isOpened()) {
        cap >> frame;
        if(frame.empty()) {
            std::cout << "Video over" << std::endl;
            break;
        }
        cv::imshow(window, frame);
    }
}

int main(int argc, char *argv[]) {
    std::thread t1(work, "/Path/to/test.mp4", "test");
    t1.join();

    // work("/Path/to/test.mp4", "test"); // it works if just call function work()

    std::cout << "Done..." << std::endl;
}

