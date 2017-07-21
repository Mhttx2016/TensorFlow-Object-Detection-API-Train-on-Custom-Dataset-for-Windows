# TensorFlow-Object-Detection-API-on-Windows
The deployment of object detection on windows
## 1. install python3.5.x
> Reference:   
> http://blog.csdn.net/zhunianguo/article/details/53524792 
## 2. install numpy,scipy, matplotlib
> (1). Download numpy.whl scipy.whl from http://www.lfd.uci.edu/~gohlke/pythonlibs/   
> (2). Install  numpy.whl scipy.whl wheel in terminal via: 

        pip3 install *.wheel  
> (3). open command prompt and make  

        pip3 install matplotlib

## 3. install tensorflow with GPU support
Reference:   
https://www.tensorflow.org/install/install_windows  
http://www.cnblogs.com/hzm12/p/6422701.html  
https://stackoverflow.com/questions/43942185/failed-to-load-the-native-tensorflow-runtime-python-3-5-2 
### 3.1 install GPU driver
> (1).look up the GPU model(eg.Gforce GTX1070) in device manager  
> (2).download corresponding driver from http://www.nvidia.cn/Download/index.aspx?lang=cn and install by double clicking .exe  

### 3.2 install CUDA Toolkit8.0  
> (1).Download CUDA8.0 from https://developer.nvidia.com/cuda-toolkit and install by double clicking .exe  
> (2).Validate the installation in cmd terminal via:  

            nvcc -V
>> get information in the picture below if success:  
![image](https://github.com/Mhttx2016/TensorFlow-Object-Detection-API-on-Windows/tree/master/pics/nvcc.png)  

> (3).Add CUDA to System Path  
>> open environment variables setting(system variable) there are already 'CUDA_PATH' and  CUDA_PATH_V8_0' exist, we need add aditional system variable:   

            CUDA_SDK_PATH = C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0 
            CUDA_LIB_PATH = %CUDA_PATH%\lib\x64 
            CUDA_BIN_PATH = %CUDA_PATH%\bin 
            CUDA_SDK_BIN_PATH = %CUDA_SDK_PATH%\bin\win64 
            CUDA_SDK_LIB_PATH = %CUDA_SDK_PATH%\common\lib\x64
>> then,add below to the end of system variable 'PATH':  
 
            ;%CUDA_LIB_PATH%;%CUDA_BIN_PATH%;%CUDA_SDK_LIB_PATH%;%CUDA_SDK_BIN_PATH%;
 >> restart your computer  

### 3.3 install cuDNN_v5.1
> (1).Download cuDNN_v5.1 from https://developer.nvidia.com/cudnn  
> (2).Copy CuDNN files to Nvidia CUDA toolkit folder  
>> when 3.2 has completed (usually is located on C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0)
>> * copy cudnn\bin\cudnn64_5.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\   
>> * copy cudnn\include\cudnn.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\   
>> * copy cudnn\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\   

### 3.4 install TensorFlow  
> (1).Install Tensorflow via pip command prompt 

            pip3 install --upgrade tensorflow-gpu
 >**NOTE:** make sure that Visual C++ Redistributate 2015 x64 is installed. If not, download it           
> (2).Validate your installation

            >>> import tensorflow as tf
            >>> hello = tf.constant('Hello, TensorFlow!')
            >>> sess = tf.Session()
            >>> print(sess.run(hello))  
>> If the system outputs the following, then you are ready to begin writing TensorFlow programs:  

            Hello, TensorFlow!
            
## 4. TensorFlow Object Detection API Installation
Reference: https://github.com/Mhttx2016/models/blob/master/object_detection/g3doc/installation.md  
### 4.1 Dependencies   
> Tensorflow Object Detection API depends on the following libraries:  
* Protobuf 2.6  
* Pillow 1.0  
* lxml  
* tf Slim (which is included in the "tensorflow/models" checkout)  
* Jupyter notebook  
* Matplotlib（completed)  
> (1).Protobuf Installation
>> Download protoc-3.3.0-win32.zip from https://github.com/google/protobuf/releases, Copy protoc.exe in directory protoc-2.6.1-win32 to C:\Windows\System32(equivalent add to system variable 'PATH'). Validate via make '>protoc' in cmd prompt.  

> (2).pillow, jupyter Installation  

            pip3 install pillow
            pip3 install jupyter
            
> (3). lxml 
>> download lxml-3.8.0-cp36-cp36m-win_amd64.whl from http://www.lfd.uci.edu/~gohlke/pythonlibs/#lxml, install via pip3 install *.whl(otherwise may cause import error when import lxml.etree)

### 4.2 Protobuf Compilation  
> The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled. This should be done by running the following command from the tensorflow/models directory:  

            # From tensorflow/models/
            protoc object_detection/protos/*.proto --python_out=.
            
### 4.3 Add Libraries to PYTHONPATH  
> When running locally, the tensorflow/models/ and slim directories should be appended to PYTHONPATH. **if there is no PYTHONPATH exist, create system variable with name=PYTHONPATH,value=tensorflow/models/;tensorflow/models/slim** (in order to import thrid party module, we need create PYTHONPATH system variable)

### 4.4 Testing the Installation
> cd to tensorflow/models/ and run  
    
            python object_detection/builders/model_builder_test.py
            
### 4.5 
