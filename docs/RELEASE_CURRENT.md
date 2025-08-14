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
