# Edge Detection with Sobel & OpenMP

👉 See [branch cuda](https://github.com/KafCoppelia/Sobel_with_OpenMP_and_CUDA/tree/cuda) for CUDA version.

## 📦 简介

基于OpenCV，使用Sobel算子实现边缘检测，并通过OpenMP实现加速计算

运行平台：gcc 12.1.0 on GNU/Linux

## 🛠️ 运行

1. 确保配置了OpenCV、OpenMP（如果是GCC则无需配置)。
2. 初次运行时，可建立 `build` 文件夹，并执行如下命令。测试图片及边缘检测结果图均位于  `./pics` 下。

```shell
mkdir build
cd build
cmake ..
make -j16
cd ..
./build/sobel ./pics/test.png
```

3. 通过修改 `CMakeLists.txt` 下 `set(USE_OMP ON)` 的 `ON/OFF` ，并重新编译，即可启用/不启用OpenMP进行加速计算。**实际测试加速效果不明显**。
4. 运行用时可能与目前CPU运行状态有关。

## 📚 参考

[ElrikPiro/sobel](https://github.com/ElrikPiro/sobel)
