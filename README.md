# COVyolo

## 说明

```bash
.
├── CMakeLists.txt # CMakeLists 重要！
├── README.md # 说明文件
├── bin # 二进制文件输出目录
├── include # include
├── lib # 第三方库，不可移除！
├── model # 模型文件
│   ├── IR_int8_1024 # 量化后1024输入模型
│   ├── IR_int8_1024_Pruned # 量化剪枝1024输入模型
│   ├── IR_int8_640 # 量化后640输入模型
│   └── IR_int8_640_Pruned # 量化剪枝640输入模型，默认
└── src # src
```

在 */bin* 中已经保存了编译后的文件可以直接使用。

需要安装libopencv-dev与OpenVINO(C++ API)的依赖。

在部分情况下可能需要重新编译（需要libopencv-dev与OpenVINO(C++ API)的依赖）：
```bash
mkdir build

cd build

cmake ..

make
```

### 参数说明

如果参数错误输入，程序会终止，并会给出Optional arguments。

```bash
Optional arguments:
  -h, --help     shows help message and exits 
  -v, --version  prints version information and exits 
  -d, --dir      Input Folder [required]
  -m, --model    model (option) [nargs=0..1] [default: "model/IR_int8_640_Pruned/last.xml"]
  -o, --out      Output Folder [required]
  --debug        (option)(output marked image) <output_labeled_image_file>(option) [nargs=0..1] [default: ""]
  --debug_IoU    <IoU_NMS>(option) [nargs=0..1] [default: 0.3]
  --debug_Cof    <Confidence_NMS>(option) [nargs=0..1] [default: 0.2]
```

1. -d, --dir 为图片输入目录，可使用相对路径。
2. -m, --model 可选的，为模型输入目录，可使用相对路径，默认为量化剪枝640模型。
3. -o, --out 为xml输出目录，可使用相对路径。
4. --debug 可选的，输出标记后的图片，便于检查。
5. --debug_IoU 可选的，NMS的IoU，default: 0.3。
6. --debug_Cof 可选的，置信度筛选，default: 0.2。

示例：
```bash
./bin/COVyolo \
--dir example/image \
--model model/IR_int8_640_Pruned/last.xml \
--out example/labels_xml \
--debug example/labeled_image \
--debug_IoU 0.3 \
--debug_Cof 0.2
```