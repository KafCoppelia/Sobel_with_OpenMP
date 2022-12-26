# Edge Detection with Sobel & CUDA

## 📦 简介

基于OpenCV，使用Sobel算子实现边缘检测，并通过CUDA实现加速计算，与CPU/OpenMP方法做对比

运行平台：gcc 7.5.0 + CUDA 10.2(1080Ti)

## 🛠️ 运行

1. 确保配置了CUDA、OpenCV、OpenMP（如果是GCC则无需配置)。
2. 初次运行时，可建立 `build` 文件夹，并执行如下命令。测试图片及边缘检测结果图均位于  `./pics` 下。

```shell
mkdir build
cd build
cmake ..
make -j16
cd ..
./build/sobel ./pics/test.png
```

3. **实际测试加速效果不明显**，可能显卡调度花费了更多的时间。

## 📚 参考

[ElrikPiro/sobel](https://github.com/ElrikPiro/sobel)
