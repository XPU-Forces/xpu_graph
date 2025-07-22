import torch
import torch_npu

import os
import re
import sys
import torch_npu._inductor
from torch_npu.contrib import transfer_to_npu
import torch._inductor.package as inductor_package

from typing import Dict, Any

import importlib
import json
import shutil
import shlex
import subprocess
import logging

from abc import ABC, abstractmethod

DEBUG_MODE = os.getenv("DEBUG", 0)

def MakePath(directory, name):
    return os.path.abspath(os.path.join(directory, name))

DEPLOY_KERNEL_PATH = "/host/deberta_files"

def modify_class_name(module_code: str) -> str:
    """ replace '<lambda>' with 'testModule' """
    modified_code = re.sub(
        r'class <lambda>\(torch\.nn\.Module\):',
        "class testModule(torch.nn.Module):",
        module_code,
        count=1
    )
    header = """
import torch
from torch import device
import torch_npu
import xpu_graph

from xpu_graph.passes.patterns.targets.npu.triton_kernel.fused_brc_permute_sum import fused_brc_permute_sum
from xpu_graph.passes.patterns.targets.npu.triton_kernel.fused_div_mul_sum import fused_div_mul_sum

import os
import torch_npu._inductor
from torch_npu.contrib import transfer_to_npu\n\n
"""
    return header + modified_code

# analysis forward func string and generate input tensors
def generate_inputs(code: str) -> Dict[str, torch.Tensor]:
    # arg0_1: "i64[11, 12, 256, 256]"
    pattern = r"(arg\d+_\d+): \"([if]\d+)\s*\[(.*?)\]\""

    # 使用正则表达式查找所有匹配项
    matches = re.findall(pattern, code)

    from torch._dynamo.testing import rand_strided
    # 解析结果
    fake_params = {}
    dtype_map = {
        "i64": torch.int64,
        "i32": torch.int32,
        "i16": torch.int16,
        "i8": torch.int8,
        "i1": torch.bool,
        "f16": torch.float16,
        "f32": torch.float32,
        "bf16": torch.bfloat16,
    }
    for match in matches:
        param_name = match[0]
        dtype = match[1]
        shape = tuple(int(dim) for dim in match[2].split(','))
        
        fake_params[param_name] = torch.zeros(shape,dtype=dtype_map[dtype],device="npu")

    return fake_params

def import_from_path(input_path):
    module_name = os.path.basename(input_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, input_path)
    if not spec:
        raise ImportError(f"can not create package: {input_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def process_and_run_model(input_path: str, do_aoti = False):
    # 1. read and replace class
    with open(input_path) as f:
        code = f.read()
    modified_code = modify_class_name(code)

    # 2. generate inputs
    forward_str =  re.search(r"def forward\(.*?\):\n", modified_code).group()
    fake_inputs = generate_inputs(forward_str)

    # 3. declare module
    # module_dict = {}
    output_path = os.path.join(os.path.dirname(input_path),"decorated_graph.py")
    with open(output_path, "w") as f:
        f.write(modified_code)    
    
    module = import_from_path(output_path)

    # 4. create a module object
    model = module.testModule().to("npu")

    # 5. run module
    with torch.no_grad():
        if do_aoti:
            exported = torch.export.export(model, tuple(fake_inputs.values()))
            output_path = torch._inductor.aoti_compile_and_package(
                exported,
                # [Optional] Specify the generated shared library path. If not specified,
                # the generated artifact is stored in your system temp directory.
                package_path=os.path.join(os.path.dirname(input_path),"origin.pt2"),
            )
        else:
            print(model(*(fake_inputs.values())))

    return output_path

class OpCodeGenerator(ABC):

    @abstractmethod
    def generate(self, num, opname, inputs, outputs):
        pass

    def get_intput_arg_str(self, node_id, opname, inputs):
        self.NewLines = ["    // PATCHED_CODE :"]
        # generate input declare
        new_input_argnames = []
        for i, (argtype, value) in enumerate(inputs):
            if argtype == "as_tensor":
                new_input = f"node{node_id}_{opname}_{i}"
                new_input_argnames.append(new_input)
                self.NewLines.append(f"    at::Tensor {new_input} = *reinterpret_cast<at::Tensor*>({value}.get());")
            elif argtype == "as_int":
                new_input_argnames.append(str(value))
            elif argtype == "as_string":
                new_input_argnames.append('"'+ value+'"')
            elif argtype == "as_float":
                new_input_argnames.append(str(value))
            else :
                raise TypeError(f"can not generate unsupport argtype: {type(arg).__name__}")

        # generate func call
        input_arg_str = ", ".join(new_input_argnames)   
        return input_arg_str

class LibraryOpGenerator(OpCodeGenerator):
    CUSTOMOP_CALLLINE_MAP={
        "broadcast_gather": "    auto {output_argname} = xpu_ops::kernels::BroadcastGather::inference({func_arg_str}); // {opname}",
        "disetangle_attention": "    auto {output_argname} = xpu_ops::kernels::DistangleAttention::inference({func_arg_str}); //{opname}",
    }
 
    def call_op_line(self, output_argname, opname, func_arg_str):
        default_line = "    auto {output_argname} = at::{opname}({func_arg_str});"
        return LibraryOpGenerator.CUSTOMOP_CALLLINE_MAP.get(opname ,default_line).format(
            output_argname = output_argname,
            opname = opname,
            func_arg_str = func_arg_str,
        )

    def generate(self, node_id, opname, inputs, outputs):
        # generate func call args
        intput_arg_str = self.get_intput_arg_str(node_id, opname, inputs)

        output_argname = f"{opname}_outs_{node_id}"
        self.NewLines.append(self.call_op_line(output_argname, opname, intput_arg_str))

        if len(outputs) == 1:
            outbuf = outputs[0]
            self.NewLines.append(f"    RAIIAtenTensorHandle {outbuf}(reinterpret_cast<AtenTensorHandle>(new at::Tensor({output_argname})));")
        else:
            for i, outbuf in enumerate(outputs):
                self.NewLines.append(f"    RAIIAtenTensorHandle {outbuf}(reinterpret_cast<AtenTensorHandle>(new at::Tensor(std::get<{i}>({output_argname}))));")

        return self.NewLines


class FallbackData:
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.nodes = json.load(f)['nodes']

    def get_node_info(self, node_id: int):
        if node_id < 0 or node_id >= len(self.nodes):
            raise ValueError(f"Invalid node_id: {node_id}. Total nodes: {len(self.nodes)}")

        # 提取目标节点
        node = self.nodes[node_id]['node']

        # 遍历所有输入参数
        arg_tuples = []
        for input_item in node['inputs']:
            arg = input_item['arg']
            arg_tuple = next(iter(arg.items()))  # as_tensor | as_float | as_int | as_string | as_float
            arg_tuples.append(arg_tuple)

        return node['target'], arg_tuples

class CodeManager:
    OP_REGISTRY = {
        # "xpu_ops::broadcast_gather": {"revertlines": 2, "generator": LibraryOpGenerator()},
        "xpu_ops::disetangle_attention": {"revertlines": 10, "generator": LibraryOpGenerator()},
        # "aten::addmm": {"revertlines": 2, "generator": LibraryOpGenerator()},
        # "aten::gather": {"revertlines": 2, "generator": LibraryOpGenerator()},
        # "aten::gelu": {"revertlines": 2, "generator": LibraryOpGenerator()},
    }

    def __init__(self, directory, cpp_name, json_name, batch_size):
        self.code_list = []
        self.cpp_path = MakePath(directory, cpp_name)
        self.batch_name = f"batch_{batch_size}"
        self.proxy_data = FallbackData(MakePath(directory, json_name))

    def clear(self):
        self.code_list.clear()
    
    def pop_lines(self, num):
        for _ in range(num):
            self.code_list.pop()

    def append_lines(self, newlines):
        self.code_list.extend(newlines)

    def regist_all_custom_ops(self):
        self.append_lines([
            "namespace xpu_ops::kernels {",
            "    struct DistangleAttention {",
            "        static std::tuple<at::Tensor, at::Tensor, at::Tensor> inference(",
            "            const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& pos_key,",
            "            const at::Tensor& pos_query, const at::Tensor& relative_pos, const at::Tensor& attn_mask,",
            "            const std::string pos_attr_type, const double score_scale);",
            "    };",
            "}  // namespace xpu_ops::kernels",
        ])

    def save_new_file(self, new_file_path):
        with open(new_file_path, "w", encoding="utf-8") as f:
            for line in self.code_list:
                f.write(line + "\n")

    def bind_proxy_data_with_real_line(self, line: str, inputs: list[tuple]):
        # 匹配 int64_t vector
        int64_pattern = r'std::vector<int64_t>\s*{\s*([^}]+?)\s*}\s*\.data\(\)'
        int64_match = re.search(int64_pattern, line)
        
        # 匹配 AtenTensorHandle vector
        aten_pattern = r'std::vector<AtenTensorHandle>\s*{\s*([^}]+?)\s*}\s*\.data\(\)'
        aten_match = re.search(aten_pattern, line)
        
        # 提取 int64_t 元素
        int64_list = []
        if int64_match:
            int64_content = int64_match.group(1)
            int64_list = [e.strip() for e in int64_content.split(',')]
        
        # 提取 AtenTensorHandle 元素
        aten_list = []
        if aten_match:
            aten_content = aten_match.group(1)
            aten_list = [e.strip() for e in aten_content.split(',')]

        arglist = []
        iptr = 0
        aptr = 0
        for argtype, refvalue in inputs:
            new_arg = [argtype, refvalue]
            if argtype == "as_tensor":
                new_arg[1] = aten_list[aptr]
                aptr+=1
            elif argtype == "as_int":
                new_arg[1] = int(int64_list[iptr])
                iptr+=1
            elif argtype == "as_string" or argtype == "as_float":
                new_arg[1] = refvalue
            else:
                raise ValueError(f"meeting unsupported argtype:{argtype}")
            arglist.append(tuple(new_arg))

        return arglist, aten_list[aptr:]

    def process_cpp_file(self):
        fallbackOpPrefixPattern = re.compile(
            r'^\s*aoti_torch_proxy_executor_call_function\(\s*proxy_executor\s*,\s*(\d+),'
        )

        compileCmdPattern = re.compile(
            r'^//\sg\+\+\s+\S+\.cpp'
        )

        linkCmdPattern = re.compile(
            r'^//\sg\+\+\s+\S+\.o'
        )

        loadKernelPattern = re.compile(
            r'loadKernel\(\"(.*/)([^/]+\.cubin)\"'
        )

        custom_op_define_header = re.compile(
            r'\#include \"experiment\/runtime\/runtime\/rt\.h\"'
        )

        kernelPathPattern = r'/tmp/(?:.*/)*([^/]+\.cubin)'

        launchKernelPattern = re.compile(
            r'launchKernel\(\"'
        )
        launch_cnt=0

        with open(self.cpp_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 保留原始行
                self.code_list.append(line.rstrip('\n'))
                
                if custom_op_define_header.search(line):
                    self.regist_all_custom_ops()

                if launchKernelPattern.search(line) and DEBUG_MODE:
                    self.code_list.append("        {")
                    self.code_list.append("            aclError error_flag = c10_npu::npuSynchronizeDevice();")
                    self.code_list.append("            if(error_flag!=ACL_SUCCESS){")
                    self.code_list.append(f"            std::cerr<<\"[DEBUG] failed to synchronize TT_kernel {launch_cnt}\"<<std::endl;")
                    self.code_list.append("            throw std::runtime_error(std::to_string(error_flag));\n            }")
                    self.code_list.append(f"            std::cerr<<\"[DEBUG] synchronized launch TT_kernel {launch_cnt}\"<<std::endl;")
                    self.code_list.append("        }")
                    launch_cnt+=1
                    continue

                if compileCmdPattern.search(line):
                    originCmd = line.replace("// ", "", 1)
                    continue
                
                if linkCmdPattern.search(line):
                    linkCmd = line.replace("// ", "", 1)
                    continue
                
                if loadKernelPattern.search(line):
                    self.pop_lines(1)
                    modified_line = re.sub(
                        kernelPathPattern,
                        rf'{MakePath(DEPLOY_KERNEL_PATH, f"{self.batch_name}/data/aotinductor/model")}/\1',
                        line
                    )
                    self.code_list.append(modified_line)
                    print(f" patched loadKernel line : {modified_line}")
                    continue

                # 检查是否匹配代理函数调用
                match = fallbackOpPrefixPattern.search(line)
                if not match:
                    continue

                # 提取 node_id 并获取算子信息
                node_id = int(match.group(1))
                target, inputs = self.proxy_data.get_node_info(node_id)
                
                # 检查算子是否已注册
                if target not in CodeManager.OP_REGISTRY:
                    continue

                inputs, outputs = self.bind_proxy_data_with_real_line(line, inputs)

                revertlines = CodeManager.OP_REGISTRY[target]["revertlines"]
                generator = CodeManager.OP_REGISTRY[target]["generator"]
                self.pop_lines(revertlines)
                print(f" Trigger patch at proxy node {node_id}, opname = {target}")
                new_lines = generator.generate(node_id, target.split("::")[-1], inputs, outputs)
                self.append_lines(new_lines)
                

        return originCmd, linkCmd


class AOTIPkgManagerNpu:
    def __init__(self,pt2_path, weight_path, new_name_prefix, batch):
        self.binfiles = []           # .cubin 
        self.wrapper_name = None     # xxx.cpp 
        self.proxy_json_name = None  # xxx.json 
        self.metadata_json_name = None  # xxx_metadata.json 
        self.weight_name = None      # .o 
        self.shared_library_name = None  # .so

        self.weight_path = weight_path
        self.batch_size = batch
        self.new_name_prefix = new_name_prefix

        self.extract_dir = self.extract_pt2(pt2_path)
        self.classify_files(self.extract_dir)
        self.new_cpp_path = MakePath(self.extract_dir, self.new_name_prefix+".cpp")
        self.new_so_path = MakePath(self.extract_dir, self.new_name_prefix+".so")

        self.code_manager = CodeManager(
            self.extract_dir,
            self.wrapper_name,
            self.proxy_json_name,
            self.batch_size,
        )

    def classify_files(self, directory):
        from pathlib import Path
        
        path = Path(directory)

        for file in path.glob("*"):
            if file.suffix == ".cubin":
                self.binfiles.append(file.name)
            elif file.suffix == ".cpp":
                self.wrapper_name = file.name
            elif file.suffix == ".json":
                if file.stem.endswith("_metadata"):
                    self.metadata_json_name = file.name
                elif file.stem.endswith("_npu"):
                    self.proxy_json_name = file.name
            elif file.suffix == ".o":
                self.weight_name = file.name
            elif file.suffix == ".so":
                self.shared_library_name = file.name
        print(f" binfiles: cnt={len(self.binfiles)}, {self.binfiles}")
        print(f" wrapper_name = {self.wrapper_name}")
        print(f" metadata_json_name = {self.metadata_json_name}")
        print(f" proxy_json_name = {self.proxy_json_name}")
        print(f" weight_name = {self.weight_name}")
        print(f" shared_library_name = {self.shared_library_name}")
        
    
    def extract_pt2(self, pt2_path: str) -> None:
        """
        unzip <path-to>/*.pt2 to <path-to>/pt2tmp
        and return <path-to>/pt2tmp/data/aotinductor/model
        """

        self.pt2_dir = os.path.dirname(pt2_path)
        extract_dir = os.path.join(self.pt2_dir, "pt2tmp")
        extract_dir = os.path.abspath(extract_dir)

        if os.path.exists(extract_dir):
            def handle_error(func, path, exc_info):
                import stat
                if not os.access(path, os.W_OK):
                    os.chmod(path, stat.S_IWUSR)
                    func(path)
                else:
                    raise
            shutil.rmtree(extract_dir, onerror=handle_error)
        os.makedirs(extract_dir, exist_ok=True)

        import zipfile
        with zipfile.ZipFile(pt2_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        return os.path.join(extract_dir,"data/aotinductor/model")

    def insert_link_libs(self, link_list, link_libs):
        if len(link_libs)==0:
            return link_list
        last_index = -1
        for i, item in enumerate(link_list):
            if item.startswith("-L"):
                last_index = i
        
        if last_index != -1:
            insert_position = last_index + 1
        
        new_list = link_list[:insert_position] + link_libs + link_list[insert_position:]
        return new_list


    def rewrite_cpp_wrapper(self, link_libs):
        old_compile_cmd, old_link_cmd = self.code_manager.process_cpp_file()
        self.code_manager.save_new_file(self.new_cpp_path)

        compile_list = shlex.split(old_compile_cmd)
        link_list = shlex.split(old_link_cmd)

        compile_list[1] = self.new_cpp_path
        tmp_path = MakePath(self.extract_dir, "tmp.o")
        compile_list[-1] = tmp_path

        link_list[1] = tmp_path
        if link_list[2].endswith(".o"):
            link_list[2] = self.weight_path
        link_list[-1] = self.new_so_path

        link_list = self.insert_link_libs(link_list, link_libs);

        print(" after rewrite_cpp_wrapper:")
        print(f" new_cpp_path = {self.new_cpp_path}")
        print(f" compile_list = {compile_list}")
        print(f" link_list = {link_list}")

        return compile_list, link_list

    def recompile(self, compile_cmd, link_cmd):
        
        try:
            subprocess.run(compile_cmd, check=True)
        except Exception as e:
            raise e

        try:
            subprocess.run(link_cmd, check=True)
        except Exception as e:
            raise e


    def repackage(self, new_pt2_directory, extra_files):
        new_proxy_json_path = MakePath(self.extract_dir, self.new_name_prefix + "_npu.json")
        new_metadata_json_path = MakePath(self.extract_dir, self.new_name_prefix + "_metadata.json")

        shutil.copy(MakePath(self.extract_dir,self.proxy_json_name), new_proxy_json_path)
        shutil.copy(MakePath(self.extract_dir, self.metadata_json_name),new_metadata_json_path)

        file_list = [
            self.new_cpp_path,
            self.new_so_path,
            new_proxy_json_path,
            new_metadata_json_path,
        ]

        for filename in self.binfiles:
            file_list.append(MakePath(self.extract_dir, filename))

        from pathlib import Path
        for extra_file in extra_files:
            try:
                Path(extra_file).resolve(strict=True)
            except Exception as e:
                raise e
            file_list.append(extra_file)

        if len(new_pt2_directory)==0:
            new_pkg_path = MakePath(self.pt2_dir, self.new_name_prefix + ".pt2")
        else:
            new_pkg_path = MakePath(new_pt2_directory, self.new_name_prefix + ".pt2")

        inductor_package.package_aoti(new_pkg_path, file_list)
        print(f" OUTPUT NEW AOTI PACKAGE TO: {new_pkg_path}")

        return new_pkg_path


    def make_new_pt2(self, new_pt2_directory="", link_libs = [], extra_files = []):
        compile_cmd, link_cmd = self.rewrite_cpp_wrapper(link_libs)
        self.recompile(compile_cmd, link_cmd)
        new_pkg_path =  self.repackage(new_pt2_directory, extra_files)
        print(f" ---------- SUCCESS MAKE NEW PT2 BATCH {self.batch_size} ----------")
        return new_pkg_path
