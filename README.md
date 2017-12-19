
# Custom optical flow Lucas Kanade Pyramidal

<div align="center">
    <img src="doc_images/sample2.png", width="450">
</div>
</div>
    This repository includes a CPU and a GPU (NVIDIA) version of the Lucas-Kanade Pyramidal flow algorithm. Besides it provides two test frameworks to test accuracy and performance of both.
    

## Files description
The following files are part of this repository:

- **lkpyramidal.cu**: GPU implementation of LK Pyramidal.
- **test_of.cpp**: Accuracy tester of the custom LK Pyramidal implementation. OpenCV and OpenPose ground truth are shown.
- **test.sh**: Script to run some basic tests and see results.
- **test_of_speed.cpp**: Performance (speed) tester of the custom LK Pyramidal implementation.
- **CMakeLists.txt**: Edit this to decide what to compile (test_of.cpp or test_of_speed.cpp)
- **CycleTimer.h**: Header file used by the performancee tester.
- **Test data directories**: ./data. ./boxing, ./street, ./lecture

</div>

