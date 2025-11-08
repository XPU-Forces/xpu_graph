# Release 0.7.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.7.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0 或者更高

## 重大变动
- Npu后端新增默认的`vendor_compiler_config`配置，此后，我们正式约定，**None表示禁用后端编译器，{}表示使用默认后端编译器**,
  此外，先前版本中，对于Npu后端，如果传入`{}`，默认将使能`inductor`，现在已经改为了`AclGraph`，对于MLU后端，我们没有变化; #444

## Bug修复与其他改动
- 修复`vendor_compiler_config`默认为`None`导致运行失败的问题; #444
