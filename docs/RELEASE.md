# Release 0.4.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0 或者更高

## 主要特性与改进
- PluginPattern现在支持字面值(`float`,`bool`和`int`)类型的继承重写，只需要将其作为Pattern函数的输入参数传入即可，有以下几点需要注意：#350
  - 作为example input传入的literal，必须确保是全局唯一的，本身也是假的无意义的，因此理论上可以选择任意值来保证唯一性；
  - 对于不需要继承重写的literal，不要作为输入传入，且需要确保有意义，特别是会影响到图节点生成；
- 现在我们通过`vendor_compiler_config`的`enable_super_kernel`来打开`super kernel`优化；#345
-

## Bug修复与其他改动
- 修复了PyTorch2.7.0下使用`MLU CppWrapper`失败的问题；#328
- 修复了`layernorm+cast`的类型转换问题; #341
- 现在我们不再对`non-inductor-based`的后端，比如`GE`和`AclGraph`，执行图分解下降，以免引入不支持的Op类型；#351

---

# Release 0.3.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0

## 重大变动
-

## 主要特性与改进
- 新增`vendor_compiler_config`字段`use_custom_pool`，用户可以通过`torch.npu.graph_pool_handle()`获取内存池并传入，从而实现多图的内存复用，但切忌并发使用共享内存池的多图，以免造成不必要的内存踩踏问题; #324

## Bug修复与其他改动
- 由于`graphviz`在处理大图时速度非常慢，现在使能`dump_graph`，我们不再以`svg`格式绘制并导出，而是以`dot`格式保存，并附上绘图代码，用户可以根据需要自行绘图，请确保绘图环境里安装了`graphviz`和python模块`pydot`；此外，我们还将图表示文件的后缀从`.ll`改成了`.txt`，这是为了能够在一些在线文档中直接预览，而无需下载查看; #286

---

# Release 0.2.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [byted-triton-x] 3.2.0

## 重大变动
- 现在whl包更名为`byted-xpu-graph`，并在bytedpypi源上发布，注意与旧版本的包名不同 #313;
- 现在`Target.ascend`被删除且合并到`Target.npu`中，`Target.ascend`和`Target.npu`原先语义，可以通过`vendor_compiler_mode:{compiler: 'ge'}`以及`vendor_compiler_mode:{compiler: 'inductor'}`区分 #274;

## 主要特性与改进
- 现在可以通过环境变量`XPUGRAPH_DEPRECATED_AOT_CONFIG_IS_EXPORT`来控制AOTConfig的`is_export`属性，这可以解决部分`immutable functional tensor`的问题，但副作用不明确，注意未来我们将弃用这一环境变量，使用该环境变量前务必明白使用的意义，否则可能会导致一些未知的问题 #311;
- 现在`Target.npu`支持对`torch.ops.npu.npu_quant_matmul`的权重进行`nd2nz`的转化，通过常量折叠完成编译期优化 #239;
- 现在`Target.npu`支持对`add+rmsNorm`和`silu+mul` 两个pattern基于triton算子进行融合，需要使能`OptLevel.level2`;
- 增加 rms_norm pattern，将常见的rms_norm小算子实现进行替换 #297;
- 现在`Target.mlu`下如果传入`vendor_compiler_mode=none`，则不会使用mlu的inductor编译器 #252;

## Bug修复与其他改动
- 在时间戳之后增加了进程与线程ID的打印，方便调试 #270;

---
