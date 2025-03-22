#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>

using namespace cv;
using namespace std;

void showProgressBar(int current, int total) {
    int barWidth = 50;
    float progress = (float)current / total;
    int pos = barWidth * progress;

    cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << int(progress * 100.0) << "%\r";
    cout.flush();
}

bool isTargetDetected(Mat& frame, Mat& target) {
    Mat result;
    matchTemplate(frame, target, result, TM_CCOEFF_NORMED);

    double minVal, maxVal;
    minMaxLoc(result, &minVal, &maxVal);

    return maxVal > 0.75;
}

int main() {
    string videoFile;
    string logFile = "detection_log.txt";
    string finalVideoWithAudio = "Honey_Pie.mp4";

    cout << "Enter the video file path: ";
    getline(cin, videoFile);

    int cutBefore, cutAfter;
    cout << "Enter cut duration before detection in seconds (recommended 30 sec): ";
    cin >> cutBefore;
    cout << "Enter cut duration after detection in seconds (recommended 10 sec): ";
    cin >> cutAfter;

    VideoCapture video(videoFile);
    if (!video.isOpened()) {
        cerr << "Error opening video file!" << endl;
        return -1;
    }

    Mat target = cv::imread("target.png", IMREAD_GRAYSCALE);
    if (target.empty()) {
        cerr << "Error: Could not load target image!" << endl;
        return -1;
    }

    int frameRate = video.get(CAP_PROP_FPS);
    int totalFrames = video.get(CAP_PROP_FRAME_COUNT);
    cout << "Frame Rate: " << frameRate << " FPS" << endl;
    cout << "Total Frames: " << totalFrames << endl;

    Mat frame;
    vector<int> detectionFrames;
    int currentFrame = 0;

    ofstream log(logFile);
    log << "Detected Frames Log:\n";

    auto startTime = chrono::high_resolution_clock::now();

    while (currentFrame < totalFrames) {
        video.set(CAP_PROP_POS_FRAMES, currentFrame); // Jump to the next second
        if (!video.read(frame)) {
            break; // Stop if we reach the end or cannot read the frame
        }

        showProgressBar(currentFrame, totalFrames);

        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        if (isTargetDetected(grayFrame, target)) {
            int detectedTime = currentFrame / frameRate;
            detectionFrames.push_back(currentFrame);
            log << "Detected at: " << detectedTime << " sec (Frame: " << currentFrame << ")\n";
        }

        currentFrame += frameRate; // Move to the next second
    }

    video.release();
    log.close();
    cout << "\nDetection Complete!\n";

    int cutFramesBefore = frameRate * cutBefore;
    int cutFramesAfter = frameRate * cutAfter;
    vector<pair<int, int>> segments;

    for (int i = 0; i < detectionFrames.size(); i++) {
        int startCut = max(0, detectionFrames[i] - cutFramesBefore);
        int endCut = min(totalFrames, detectionFrames[i] + cutFramesAfter);

        if (segments.empty() || startCut > segments.back().second) {
            segments.push_back({ startCut, endCut });
        }
        else {
            segments.back().second = endCut;
        }
    }

    ostringstream ffmpegFilter;
    for (size_t i = 0; i < segments.size(); i++) {
        double startSec = static_cast<double>(segments[i].first) / frameRate;
        double endSec = static_cast<double>(segments[i].second) / frameRate;

        if (i > 0) ffmpegFilter << " + ";
        ffmpegFilter << "between(t," << fixed << setprecision(2) << startSec
            << "," << fixed << setprecision(2) << endSec << ")";
    }

    string trimCommand = "ffmpeg -hwaccel cuda -i \"" + videoFile + "\" -filter_complex \""
        "[0:v]select='" + ffmpegFilter.str() + "',setpts=N/FRAME_RATE/TB[v];"
        "[0:a]aselect='" + ffmpegFilter.str() + "',asetpts=N/SR/TB[a]\" "
        "-map \"[v]\" -map \"[a]\" -c:v h264_nvenc -preset slow -cq 23 "
        "-c:a aac -b:a 128k -strict experimental -y \"" + finalVideoWithAudio + "\"";


    cout << "\nTrimming video and audio together..." << endl;
    if (system(trimCommand.c_str()) != 0) {
        cerr << "Error trimming video & audio!" << endl;
        return -1;
    }

    cout << "\nProcessing complete! Final video with audio saved as " << finalVideoWithAudio << endl;
    cout << "Detection log saved as " << logFile << endl;

    return 0;
}
