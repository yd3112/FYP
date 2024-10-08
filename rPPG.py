import numpy as np
import cv2
import sys
from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import time
import matplotlib.pyplot as plt

realWidth = 640   
realHeight = 480  
videoWidth = 160
videoHeight = 120
videoChannels = 3
videoFrameRate = 15

# Webcam Parameters
webcam = cv2.VideoCapture(0)
detector = FaceDetector()

webcam.set(3, realWidth)
webcam.set(4, realHeight)

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Plotting Setup
plt.ion()  # Interactive mode on for live plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
bpm_plot_data = []
time_data = []
frequency_data = []
red_channel_data = []
green_channel_data = []
blue_channel_data = []

# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    # Resize filteredFrame to match detectionFrame size
    filteredFrame = cv2.resize(filteredFrame, (detectionFrame.shape[1], detectionFrame.shape[0]))
    return filteredFrame

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 10
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

i = 0
ptime = 0
ftime = 0
time_counter = 0

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    frame, bboxs = detector.findFaces(frame, draw=False)
    frameDraw = frame.copy()
    ftime = time.time()
    fps = 1 / (ftime - ptime)
    ptime = ftime

    cv2.putText(frameDraw, f'FPS: {int(fps)}', (30, 440), 0, 1, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']
        
        # Include the whole face in detectionFrame
        face_x_start = x1
        face_x_end = x1 + w1
        face_y_start = y1
        face_y_end = y1 + h1
        detectionFrame = frame[face_y_start:face_y_end, face_x_start:face_x_end]

        # Define areas of interest: forehead and cheeks
        forehead_height = h1 // 4  # Forehead height
        cheek_height = h1 // 4     # Cheek height
        cheek_width = w1 // 3      # Cheek width

        # Forehead coordinates
        forehead_y_start = y1
        forehead_y_end = y1 + forehead_height
        forehead_x_start = x1
        forehead_x_end = x1 + w1

        # Left cheek coordinates
        left_cheek_y_start = y1 + forehead_height
        left_cheek_y_end = left_cheek_y_start + cheek_height
        left_cheek_x_start = x1
        left_cheek_x_end = x1 + cheek_width

        # Right cheek coordinates
        right_cheek_y_start = y1 + forehead_height
        right_cheek_y_end = right_cheek_y_start + cheek_height
        right_cheek_x_start = x1 + w1 - cheek_width
        right_cheek_x_end = x1 + w1

        # Draw rectangles on frameDraw to mark regions
        cv2.rectangle(frameDraw, (forehead_x_start, forehead_y_start), (forehead_x_end, forehead_y_end), (0, 255, 0), 2)
        cv2.rectangle(frameDraw, (left_cheek_x_start, left_cheek_y_start), (left_cheek_x_end, left_cheek_y_end), (255, 0, 0), 2)
        cv2.rectangle(frameDraw, (right_cheek_x_start, right_cheek_y_start), (right_cheek_x_end, right_cheek_y_end), (255, 0, 0), 2)

        # Resize detectionFrame to fit expected dimensions
        detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))

        # Extract and Store RGB Channel Means
        red_mean = np.mean(detectionFrame[:, :, 2])  # Red channel
        green_mean = np.mean(detectionFrame[:, :, 1])  # Green channel
        blue_mean = np.mean(detectionFrame[:, :, 0])  # Blue channel
        red_channel_data.append(red_mean)
        green_channel_data.append(green_mean)
        blue_channel_data.append(blue_mean)

        # Construct Gaussian Pyramid
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]

        # Perform Fourier Transform
        fourierTransform = np.fft.ifft(videoGauss, axis=0)

        # Bandpass Filter
        fourierTransform[mask == False] = 0

        # Amplify
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        # Reconstruct Resulting Frame
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        bufferIndex = (bufferIndex + 1) % bufferSize

        # Show the amplified face region in the corner
        outputFrame_show = cv2.resize(outputFrame, (videoWidth // 2, videoHeight // 2))
        frameDraw[0:videoHeight // 2, (realWidth - videoWidth // 2):realWidth] = outputFrame_show

        # Grab a Pulse
        if bufferIndex % bpmCalculationFrequency == 0:
            i += 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

            # Update plot data for BPM and Frequency vs Time
            current_time = time_counter  # Capture the current time to ensure time_data matches
            bpm_plot_data.append(bpm)
            frequency_data.append(hz)
            time_data.append(current_time)

            # Live plotting
            ax1.clear()
            ax1.plot(time_data, bpm_plot_data, label='BPM over time', color='b')
            ax1.set_title('Heart Rate (BPM)')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('BPM')
            ax1.legend()
            ax1.grid()

            ax2.clear()
            ax2.plot(time_data, frequency_data, label='Frequency over time (1-2 Hz)', color='r')
            ax2.set_title('Frequency over Time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Frequency (Hz)')
            ax2.set_ylim(minFrequency, maxFrequency)
            ax2.legend()
            ax2.grid()

            ax3.clear()
            ax3.plot(time_data, red_channel_data[:len(time_data)], label='Red Channel', color='r')
            ax3.plot(time_data, green_channel_data[:len(time_data)], label='Green Channel', color='g')
            ax3.plot(time_data, blue_channel_data[:len(time_data)], label='Blue Channel', color='b')
            ax3.set_title('RGB Channel Means Over Time')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Channel Mean Intensity')
            ax3.legend()
            ax3.grid()

            plt.pause(0.01)  # Pause to update the plot

        bpm_value = bpmBuffer.mean()

        if i > bpmBufferSize:
            cvzone.putTextRect(frameDraw, f'BPM: {bpm_value:.2f}', (videoWidth // 2, 40), scale=2)
        else:
            cvzone.putTextRect(frameDraw, "Calculating BPM...", (30, 40), scale=2)

        cv2.imshow("Heart Rate Monitor", frameDraw)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.imshow("Heart Rate Monitor", frameDraw)

    # Increment time counter after every frame processed
    time_counter += 1

# Cleanup
webcam.release()
cv2.destroyAllWindows()
plt.ioff()  # Turn off interactive mode
plt.show()  # Show final plots̀