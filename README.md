# TensorFlow-Object-Detection-API-on-Windows
The deployment of object detection on windows
## 1. install python3.5.x
> Reference:   
> http://blog.csdn.net/zhunianguo/article/details/53524792 
## 2. install numpy,scipy, opencv
> (1). Download numpy.whl scipy.whl, opencv.whl from http://www.lfd.uci.edu/~gohlke/pythonlibs/   
> (2). Install *.wheel in terminal via: 

    pip3 install *.wheel
## 3. install tensorflow with GPU support
Reference:   
https://www.tensorflow.org/install/install_windows  
http://www.cnblogs.com/hzm12/p/6422701.html  
https://stackoverflow.com/questions/43942185/failed-to-load-the-native-tensorflow-runtime-python-3-5-2 
> 3.1 install GPU driver
>> look up the GPU model(eg.Gforce GTX1070) in device manager
>> download corresponding driver from http://www.nvidia.cn/Download/index.aspx?lang=cn and install by double clicking .exe  

> 3.2 install CUDA Toolkit8.0  
>> Download CUDA8.0 from https://developer.nvidia.com/cuda-toolkit and install by double clicking .exe  
>> Verify the installation in cmd terminal via:  

    nvcc -V
>> get information in the picture below if success:  
![image](https://github.com/Mhttx2016/TensorFlow-Object-Detection-API-on-Windows/tree/master/pics/nvcc.png)  
>> Add CUDA to System Path  
>> open environment variables setting(system variable) there are already 'CUDA_PATH' and  CUDA_PATH_V8_0' exist, we need add aditional system variable:   

     CUDA_SDK_PATH = C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0 
     CUDA_LIB_PATH = %CUDA_PATH%\lib\x64 
     CUDA_BIN_PATH = %CUDA_PATH%\bin 
     CUDA_SDK_BIN_PATH = %CUDA_SDK_PATH%\bin\win64 
     CUDA_SDK_LIB_PATH = %CUDA_SDK_PATH%\common\lib\x64
>> add below to the end of system variable 'PATH':  
 
     ;%CUDA_LIB_PATH%;%CUDA_BIN_PATH%;%CUDA_SDK_LIB_PATH%;%CUDA_SDK_BIN_PATH%;
 >> then restart your computer
