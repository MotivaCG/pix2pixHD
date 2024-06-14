import os
import sys
from random import randint
import numpy as np

try:
    from PIL import Image
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    import argparse
except ImportError as err:
    sys.stderr.write(f"ERROR: failed to import module ({err})\n"
                     "Please make sure you have pycuda and the example dependencies installed.\n"
                     "https://wiki.tiker.net/PyCuda/Installation/Linux\n"
                     "pip(3) install tensorrt[examples]\n")
    exit(1)

try:
    import tensorrt as trt
    try:
        from tensorrt import OnnxParser as onnxparser
        print("new onnx")
    except ImportError as e:
        from tensorrt.parsers import onnxparser   
        print("classic onnx")
except ImportError as err:
    sys.stderr.write(f"ERROR: failed to import module ({err})\n"
                     "Please make sure you have the TensorRT Library installed\n"
                     "and accessible in your LD_LIBRARY_PATH\n")
    exit(1)

print("Create logger!")
G_LOGGER = trt.Logger(trt.Logger.Severity.INFO)

class Profiler(trt.IProfiler):
    def __init__(self, timing_iter):
        self.timing_iterations = timing_iter
        self.profile = []

    def report_layer_time(self, layerName, ms):
        record = next((r for r in self.profile if r[0] == layerName), (None, None))
        if record == (None, None):
            self.profile.append((layerName, ms))
        else:
            self.profile[self.profile.index(record)] = (record[0], record[1] + ms)

    def print_layer_times(self):
        totalTime = 0
        for i in range(len(self.profile)):
            print("{:40.40} {:4.3f}ms".format(self.profile[i][0], self.profile[i][1] / self.timing_iterations))
            totalTime += self.profile[i][1]
        print("Time over all layers: {:4.2f} ms per iteration".format(totalTime / self.timing_iterations))

def get_input_output_names(trt_engine):
    
    import tensorrt as trt
    ntensors = trt_engine.num_io_tensors
    maps = []

    for b in range(ntensors):
        name = trt_engine.get_tensor_name(b)
        dims = trt_engine.get_tensor_shape(name)
        dtype = trt_engine.get_tensor_dtype(name)
        
        if trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            maps.append(name)
            print("Found input: ", name)
        elif trt_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT :
            maps.append(name)
            print("Found output: ", name)

        print("shape=", dims)
        print("dtype=", dtype)
    return maps


def get_tensor_index(engine, desiredName):
    import tensorrt as trt
    ntensors = engine.num_io_tensors
    for b in range(ntensors):
        name = engine.get_tensor_name(b)
        if name == desiredName:
            return b
    return -1

def create_memory(engine, name, buf, mem, batchsize, inp, inp_idx):
    
    import tensorrt as trt
    print("create_memory for:"+name)
    tensor_idx = get_tensor_index(engine, name)

    if tensor_idx == -1:
        raise AttributeError("Not a valid tensor id")
    print("Tensor: name={}, tensor_idx={}".format(name, str(tensor_idx)))
    dims = engine.get_tensor_shape(name)
    eltCount = np.prod(dims) * batchsize

    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        h_mem = inp[inp_idx]
        inp_idx = inp_idx + 1
    else:
        h_mem = np.random.uniform(0.0, 255.0, eltCount).astype(np.float32)

    d_mem = cuda.mem_alloc(eltCount * 4)
    cuda.memcpy_htod(d_mem, h_mem)
    buf.insert(tensor_idx, int(d_mem))
    mem.append(d_mem)
    return inp_idx

def time_inference(engine, batch_size, inp):
    tensors = []
    mem = []
    inp_idx = 0
    for io in get_input_output_names(engine):
        inp_idx = create_memory(engine, io, tensors, mem, batch_size, inp, inp_idx)

    context = engine.create_execution_context()
    g_prof = Profiler(500)
    context.profiler = g_prof
    for _ in range(500):
        context.execute_v2(tensors)
    g_prof.print_layer_times()

def convert_to_datatype(v):
    if v == 8:
        return trt.DataType.INT8
    elif v == 16:
        return trt.DataType.HALF
    elif v == 32:
        return trt.DataType.FLOAT
    else:
        print("ERROR: Invalid model data type bit depth: " + str(v))
        return trt.DataType.INT8

def run_trt_engine(engine_file, bs, inp):
    with open(engine_file, "rb") as f:
        runtime = trt.Runtime(G_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    time_inference(engine, bs, inp)

def run_onnx_old(onnx_file, data_type, bs, inp):
    builder = trt.Builder(G_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = onnxparser(network, G_LOGGER)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if data_type == trt.DataType.HALF:
        config.set_flag(trt.BuilderFlag.FP16)
    elif data_type == trt.DataType.INT8:
        config.set_flag(trt.BuilderFlag.INT8)

    engine = builder.build_engine(network, config)
    time_inference(engine, bs, inp)    

def run_onnx(onnx_file, data_type, bs, inp_name):
    print("onnx_file:" + onnx_file) 
    import tensorrt as trt
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(G_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, G_LOGGER)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if data_type == trt.DataType.HALF:
        config.set_flag(trt.BuilderFlag.FP16)
    elif data_type == trt.DataType.INT8:
        config.set_flag(trt.BuilderFlag.INT8)

    # Use the new method to build the engine
    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        print("Failed to build serialized engine")
        return

    runtime = trt.Runtime(G_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    # Assuming you have a time_inference function that can use this engine and the name of the input tensor
    time_inference(engine, bs, inp_name)

