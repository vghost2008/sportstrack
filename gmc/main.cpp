#include <string>
#include <iostream>
#include <sstream>
#include <iterator>
#include <numeric>
#include <algorithm>
#include <experimental/filesystem>
#include <future>
#include <list>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/videostab.hpp"

using namespace std;
namespace fs=std::experimental::filesystem;
// --------------------------------------------
// Global Parameters
// --------------------------------------------
int downscale = 2;

bool WITH_MASKING = false; // true | false
// --------------------------------------------


int GMC(const string& path);
int main(int argc,char** argv)
{
    if(argc==0) {
        cout<<"./gmc dir0 dir1"<<endl;
        return -1;
    }

    list<future<int>> futures;
    constexpr auto kThreadNr = 10;

    for(auto i=1; i<argc; ++i) {
        futures.push_back(std::async(std::launch::async,GMC,argv[i]));
        if(futures.size()>=kThreadNr)
            futures.pop_front();
    }

    futures.clear();

    return 0;
}

int GMC(const string& _path)
{
    cout<<"Process "<<_path<<endl;
    string path = (fs::path(_path)/fs::path("img1")).string();
    std::vector<std::string> images;
    for (const auto & entry : std::experimental::filesystem::directory_iterator(path)) {
        auto path = entry.path();
        if(path.string().find("jpg")==string::npos)
            continue;
        images.push_back(entry.path());
    }

    std::sort(images.begin(), images.end());

    std::ifstream detFile;

    if (WITH_MASKING) {
        fs::path det_path(path);
        std::string fullDetectionsPath = (det_path/"det.txt").string();
        detFile.open(fullDetectionsPath);

        if (!detFile.is_open()) {
            std::cout << "ERROR: Unable to open detection file: " << fullDetectionsPath << std::endl;
            return -2;
        }
    }

    std::string detLine;
    int detFrameNumber = -1;
    cv::Rect detRect;
    float detScore;

    int numFrames = (int)images.size();
    cv::Mat fullFrame, frame, prevFrame;

    cv::Ptr<cv::videostab::MotionEstimatorRansacL2> est = cv::makePtr<cv::videostab::MotionEstimatorRansacL2>(cv::videostab::MM_SIMILARITY);
    cv::Ptr<cv::videostab::KeypointBasedMotionEstimator> kbest = cv::makePtr<cv::videostab::KeypointBasedMotionEstimator>(est);


    std::ofstream outFile;

    fs::path save_path(path);
    save_path /= "gmc.txt";

    outFile.open(save_path.string());

    double overallTime = 0.0;
    cv::TickMeter timer;

    for (int i = 0; i < numFrames; ++i) {
        cout<<images[i]<<endl;
        fullFrame = cv::imread(images[i], 0);

        if (fullFrame.empty())
        {
            std::cout << "ERROR: Empty frame " << images[i] << std::endl;
            continue;
        }

        timer.reset();
        timer.start();

        cv::Size downSize(fullFrame.cols / downscale, fullFrame.rows / downscale);
        cv::resize(fullFrame, frame, downSize, cv::INTER_LINEAR);

        cv::Mat warp = cv::Mat::eye(3, 3, CV_32F);

        if (!prevFrame.empty()) {
            if (WITH_MASKING)
            {
                cv::Mat mask(frame.size(), CV_8U);
                mask.setTo(255);

                // Mask detections
                if (detFrameNumber == i)
                {
                    if (detScore > 0.5)
                        mask(detRect) = 0;
                }

                while(std::getline(detFile, detLine))
                {
                    std::stringstream ss(detLine);
                    std::vector<std::string> tokens;

                    while( ss.good() )
                    {
                        std::string substr;
                        getline( ss, substr, ',' );
                        tokens.push_back(substr);
                    }

                    detFrameNumber = std::stoi(tokens[0]) - 1;
                    detRect.x = int(std::stof(tokens[1]) / downscale);
                    detRect.y = int(std::stof(tokens[2]) / downscale);
                    detRect.width = int(std::stof(tokens[3]) / downscale);
                    detRect.height = int(std::stof(tokens[4]) / downscale);
                    detScore = std::stof(tokens[6]);

                    detRect.x = std::max(0, detRect.x);
                    detRect.y = std::max(0, detRect.y);
                    detRect.width = std::min(detRect.width, frame.cols - detRect.x);
                    detRect.height = std::min(detRect.height, frame.rows - detRect.y);

                    if (detFrameNumber > i)
                        break;

                    if (detScore > 0.5)
                        mask(detRect) = 0;
                }
                if (i % 100 == 0)
                {
                    cv::namedWindow("mask", cv::WINDOW_NORMAL);
                    cv::imshow("mask", mask);
                    cv::waitKey(0);
                }

                kbest->setFrameMask(mask);
            }

            bool ok;
            warp = kbest->estimate(prevFrame, frame, &ok);


            if (!ok)
            {
                std::cout << "WARNING: Warp not ok" << std::endl;
            }

            warp.convertTo(warp, CV_32F);
            warp.at<float>(0, 2) *= downscale;
            warp.at<float>(1, 2) *= downscale;
        }

        // Store last frame
        frame.copyTo(prevFrame);

        timer.stop();
        overallTime += timer.getTimeMilli();

        // Write result to file
        std::string line = std::to_string(i) + "\t" +
                std::to_string(warp.at<float>(0, 0)) + "\t" +
                std::to_string(warp.at<float>(0, 1)) + "\t" +
                std::to_string(warp.at<float>(0, 2)) + "\t" +
                std::to_string(warp.at<float>(1, 0)) + "\t" +
                std::to_string(warp.at<float>(1, 1)) + "\t" +
                std::to_string(warp.at<float>(1, 2)) + "\t";

        std::cout << line << std::endl;
        outFile << line << std::endl;
    }

    outFile.close();

    std::cout << "GMC time [mSec]: " << overallTime / numFrames << std::endl;
    std::cout << "Saved GMC to " << save_path<< std::endl;

    return 0;
}
