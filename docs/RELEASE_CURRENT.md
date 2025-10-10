# Release 0.5.2

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0 或者更高

## 重大变动
-

## 主要特性与改进
- 现在你可以通过结构化字典的方式，将npu compiler的选项设置透传到过去，例如：
  - ```
    vendor_cfg = {'mode': 'reduce-overhead', 'aclgraph_config': {'use_custom_pool': mem_pool_handl}}
    xpu_cfg =  XpuGraphConfig(
                is_training=False,
                target=Target.npu,
                vendor_compiler_config=vendor_cfg)
    # This also works for now and will be deprecated after THREE versions.
    vendor_cfg = {'mode': 'reduce-overhead', 'use_custom_pool': mem_pool_handl}
    xpu_cfg =  XpuGraphConfig(
                is_training=False,
                target=Target.npu,
                vendor_compiler_config=vendor_cfg)

    ```
  目前我们仍然兼容原来的选项写法，但接下来三个版本后，我们将废弃原有的`use_custom_pool`字段，并不再新增npu compiler选项字段，请参考官方的选项用法。 #399
- 增加对横向融合pattern的支持，可以将多个无依赖的pointwise算子融合成一次调用，减少host开销。 #390
- 增加AddN pattern，将多次add融合为stack sum。 #248

## Bug修复与其他改动
-
