# Release 0.9.0

## 主要依赖库版本描述
- python 3.9 或者更高
- [torch-mlu] 2.7.1；
- [torch-npu] 2.7.1；
- pytorch与torch-mlu或torch-npu对应；
- [triton-x] 3.2.0 或者更高；

## 重大变动
- 解耦重构了cache机制和编译产物封装机制。 #450
  * 定义了 `SerializableArtifact` 类，所有后端如果需要产物缓存，都需要将编译产物继承自该类，并实现对应的 `_serialize` 和 `_deserialize` 方法。
  * 实现了 `BoxedCallWrapper` 类，用于封装inference compiler编译后的函数，确保在推理场景下，运行时参数能够正确传递给boxed_func形式的编译产物。

## 主要特性与改进
-

## Bug修复与其他改动
-
