# Release 0.2.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton] 3.2.0
- [triton-npu] 与triton保持一致

## 重大变动
-

## 主要特性与改进
- 新增`vendor_compiler_config`字段`use_custom_pool`，用户可以通过`torch.npu.graph_pool_handle()`获取内存池并传入，从而实现多图的内存复用，但切忌并发使用共享内存池的多图，以免造成不必要的内存踩踏问题;

# Bug修复与其他改动
- 由于`graphviz`在处理大图时速度非常慢，现在使能`dump_graph`，我们不再以`svg`格式绘制并导出，而是以`dot`格式保存，并附上绘图代码，用户可以根据需要自行绘图，请确保绘图环境里安装了`graphviz`和python模块`pydot`；此外，我们还将图表示文件的后缀从`.ll`改成了`.txt`，这是为了能够在一些在线文档中直接预览，而无需下载查看;
