# Release 0.6.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
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
- 支持训练阶段指定计算图前反向拆分策略，可以通过参数`partition_fn`指定，也可以通过环境变量`XPUGRAPH_PARTITIONER`指定。 #430
    * `"DEFAULT"`: 使用与eager行为一致的前反向拆分（`torch._functorch.partitioners.default_partition_fn`）
    * `"MINCUT"`: 使用显存优化的前反向拆分 (`torch._functorch.partitioners.min_cut_rematerialization_partition`)
    * 也可以指定自定义的partition_fn实现定制的前反向拆分策略（通过环境变量指定时使用函数的fqn进行指定）
    * 未指定时效果同`"DEFAULT"`


## Bug修复与其他改动
-
