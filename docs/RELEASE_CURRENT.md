# Release 0.8.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.7.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0 或者更高

## 重大变动
- Fallback Legacy Dispatch机制默认开启。在：
    1. 训练/推理场景输入存在subclass tensor （如`DTensor`）；或
    2. 训练场景存在higher-order operator时（如`aotograd.Function.apply`），

    均会自动回退到 aot_autograd 的编译流程，以覆盖现有dispatch机制不支持的场景
-

## 主要特性与改进
-


## Bug修复与其他改动
-
