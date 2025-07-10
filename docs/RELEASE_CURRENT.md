# Release 0.1.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应

## 主要特性与改进
- 现在可以通过环境变量`XPUGRAPH_DEPRECATED_AOT_CONFIG_IS_EXPORT`来控制AOTConfig的`is_export`属性，这可以解决部分`immutable functional tensor`的问题，但副作用不明确，注意未来我们将弃用这一环境变量，使用该环境变量前务必明白使用的意义，否则可能会导致一些未知的问题 #311;
- 现在whl包更名为`byted-xpu-graph`，并在bytedpypi源上发布，注意与旧版本的包名不同 #313;
- 现在`Target.npu`支持对`torch.ops.npu.npu_quant_matmul`的权重进行`nd2nz`的转化，通过常量折叠完成编译期优化 #239;
- 添加 add+rmsNorm和silu+mlu 两个triton 写的融合算子及 fx pattern
- 增加 rms_norm pattern，将常见的rms_norm小算子实现进行替换 #297;
-

## 相关模块API的重大变动
-

## 弃用特性声明
-

## Bug修复与其他改动
- 在时间戳之后增加了进程与线程ID的打印，方便调试 #270;
