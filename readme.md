# 涉及到的优化技术
- 内存布局优化
- 内存访问优化
- 循环展开
- SIMD
- 矩阵分块
- OpenMP
- CUDA
- CUDA+共享内存

# 运行结果比对
- 测试条件：
	- 两个1024x1024尺寸的float型矩阵相乘
	- 处理器：i5-10210U
	- RAM：8.00GB
	- 显卡：NVIDIA GeForce MX330
- 测试结果：
	- multi1 execution time: 6772ms
	- multi2 execution time: 708ms
	- multi3 execution time: 460ms
	- multi4 execution time: 136ms
	- multi5 execution time: 320ms
	- multi6 execution time: 136ms
	- multi7 execution time: 104ms
	- multi8 execution time: 103ms
	- multi9 execution time: 57ms
	- multi10 execution time: 54ms
	- multi11 execution time: 18ms
	- multi12 execution time: 306ms
	- multi13 execution time: 75ms
	- multi14 execution time: 75ms
	- multi15 execution time: 76ms
	- multi16 execution time: 71ms
	- multi17 execution time: 114ms
	- multi18 execution time: 91ms

# 依赖库
- spdlog
- cuda
- catch2
- openBlas