## Autonomous RC Car 
## ME384 Mechatronics Course Project
## Bilkent University, Ankara, Turkey

   An autonomous RC car which can stay between white lines and react when it sees a stop sign. 
   <a href="https://www.youtube.com/watch?v=-MaZoxNlZlw
" target="_blank"><img src="http://img.youtube.com/vi/-MaZoxNlZlw/0.jpg" width="360" height="240" border="10" /></a>

### Group Members
* Buğra Beytullahoğlu
* Uzay Tefek
* Merve Çetiner
* Elif Nur Özbek
* Onuralp Şimşek
* Gürcan Cengiz

### Tools and Libraries
* Raspbery Pi 3
* Raspberry Pi Camera Module
* PiCamera
* Numpy
* OpenCV
* Keras
* Tensorflow

### Codes
* HumanDrive.py: Constantly saves a frame apply image processing to it (Canny Edge Detection), and saves user input at the same time for further neural network training
* NeuralNetwork.py: Using keras and tensorflow, build a neural network model with data which was collected in HumanDrive.py
* AutonomousDriving.py: Using trained model, predicts driving conditions.

### Notes
* As junior mechanical engineering students, me and my group members had little knowledge about neural networks and image processing. Therefore, we spent a lot of
time to understand what other projects did and we were inspired by them.
* In neural network part, we played with layer numbers, layer sizes, activation functions and etc. to get a precise model.
* Our stop sign classifier was taken from an other project. I mentioned that project in references 
* You need to edit file directories, servo motor angles and pwm cycles if you need to use this project's source code

### References and Inspirations
* https://github.com/hamuchiwa/AutoRCCar
* https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
* https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/

I would be glad to get any comments and questions from anyone.
