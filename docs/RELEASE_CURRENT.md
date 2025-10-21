# Release 0.6.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0 或者更高

## 重大变动
-

## 主要特性与改进
- 支持训练阶段指定partition_fn，可以通过参数`partition_fn`或环境变量`XPUGRAPH_PARTITIONER`指定。 #430

## Bug修复与其他改动
-
