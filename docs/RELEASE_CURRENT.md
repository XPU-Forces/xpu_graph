# Release 0.10.0

## 主要依赖库版本描述
- python 3.9 或者更高
- [torch-mlu] 2.7.1；
- [torch-npu] 2.7.1；
- pytorch与torch-mlu或torch-npu对应；
- [triton-x] 3.2.0.post9 或者更高；

## 重大变动
- Fallback Legacy Dispatch机制默认开启。在：
    1. 训练/推理场景输入存在subclass tensor （如`DTensor`）；或
    2. 训练场景存在higher-order operator时（如`aotograd.Function.apply`），

    均会自动回退到 aot_autograd 的编译流程，以覆盖现有dispatch机制不支持的场景

## 主要特性与改进
- XpuGraphConfig增加调试用选项`include_patterns`和`skip_patterns`，用于额外打开或关闭特定pattern。 #463
- 调整`CombinePointwiseSource`和`CombinePointwiseSink`两个pattern的应用优化级别为`level2`。 #470

## Bug修复与其他改动
- 修复**常量折叠**相关pass错误折叠有**副作用（side-effect）**节点的问题；
- 修复mlu推理模式下触发FallbackLegacyDispatch后产物无法被上游组件正确序列化的问题；
