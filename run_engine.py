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
        print("new onnx mode")
    except ImportError as e:
        from tensorrt.parsers import onnxparser   
        print("classic onnx mode")
except ImportError as err:
    sys.stderr.write(f"ERROR: failed to import module ({err})\n"
                     "Please make sure you have the TensorRT Library installed\n"
                     "and accessible in your LD_LIBRARY_PATH\n")
    exit(1)

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
    import numpy as np
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt

    print(f"Initializing memory creation for tensor: {name}")

    try:
        tensor_idx = get_tensor_index(engine, name)
        if tensor_idx == -1:
            raise ValueError(f"Tensor name {name} not found in engine.")
        else:
            print(f"Tensor index for {name}: {tensor_idx}")

    except Exception as e:
        print(f"Error retrieving tensor index for {name}: {e}")
        return inp_idx  # Return current index to prevent further processing

    try:
        dims = engine.get_tensor_shape(name)
        print(f"Dimensions for tensor {name}: {dims}")
        eltCount = np.prod(dims) * batchsize
        print(f"Total elements (eltCount) to allocate for {name}: {eltCount}")

    except Exception as e:
        print(f"Error calculating elements count for {name}: {e}")
        return inp_idx

    try:
        h_mem = 0
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            h_mem = inp[inp_idx]
            inp_idx += 1
            print(f"Assigned host memory from input for tensor {name}.")
        else:
            h_mem = np.random.uniform(0.0, 255.0, int(eltCount)).astype(np.float32)
            print(f"Assigned host memory from random values for tensor {name}.")
            
        # Ensure h_mem is a numpy array and appropriate for memory operations
        if isinstance(h_mem, np.ndarray):
            print("valid array")
        else:
            print(f"Type of h_mem: {type(h_mem)}. Attempting conversion if possible.")
            if hasattr(h_mem, 'numpy'):  # This checks if h_mem is a tensor object with a .numpy() method
                h_mem = h_mem.numpy()
            else:
                print("Failed to convert h_mem to a numpy array.")
                return inp_idx            

        byte_size = int(eltCount * h_mem.itemsize)
        d_mem = cuda.mem_alloc(byte_size)
        cuda.memcpy_htod(d_mem, h_mem)
        buf.insert(tensor_idx, int(d_mem))
        mem.append(d_mem)

    except cuda.LogicError as e:
        print(f"CUDA logic error while allocating memory for tensor {name}: {e}")
    except cuda.MemoryError as e:
        print(f"CUDA memory error while allocating memory for tensor {name}: {e}")
    except Exception as e:
        print(f"Unhandled error during memory allocation for tensor {name}: {e}")

    return inp_idx

class Profiler(trt.IProfiler):
    def __init__(self, timing_iter):
        super().__init__()  # Inicializa correctamente la clase padre
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
        print("Tiempo total sobre todas las capas: {:4.2f} ms por iteración".format(totalTime / self.timing_iterations))

def time_inference_multi(engine, batch_size, inp):
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
    
    
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch

class TensorOutput:
    def __init__(self):
        self.data = []  # Esta lista contendrá tensores de PyTorch

    def add_tensor(self, tensor):
        self.data.append(tensor)

def time_inference(engine, batch_size, inp):
    import numpy as np
    import torch
    import pycuda.driver as cuda

    tensors = []
    mem = []
    inp_idx = 0
    tensor_output = TensorOutput()  # Usamos la nueva clase para almacenar los tensores

    for io in get_input_output_names(engine):
        inp_idx = create_memory(engine, io, tensors, mem, batch_size, inp, inp_idx)

    context = engine.create_execution_context()
    execution_times = 1  # Profiling only once
    g_prof = Profiler(execution_times)
    context.profiler = g_prof

    # Execute the inference
    for _ in range(execution_times):
        context.execute_v2(tensors)

    # Identifying number of output tensors
    output_tensors = []
    ntensors = engine.num_io_tensors
    for b in range(ntensors):
        name = engine.get_tensor_name(b)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            output_tensors.append(name)

    # Convert output data from device to PyTorch tensors and store in TensorOutput
    for name in output_tensors:
        dims = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = np.prod(dims)
        host_buffer = np.empty(size, dtype=dtype)

        tensor_idx = get_tensor_index(engine, name)
        device_allocation = mem[tensor_idx]

        cuda.memcpy_dtoh(host_buffer, device_allocation)
        reshaped_buffer = host_buffer.reshape(dims)
        pytorch_tensor = torch.from_numpy(reshaped_buffer).to(torch.float32)
        
        # Asegurarse de que el tensor esté en el formato correcto (C, H, W)
        if pytorch_tensor.dim() == 4:  # Batch size presente
            pytorch_tensor = pytorch_tensor[0]  # Tomar el primer elemento del batch
        
        if pytorch_tensor.dim() == 3 and pytorch_tensor.shape[0] in [1, 3]:  # Check if already (C, H, W)
            pass  # El tensor ya está en el formato correcto
        elif pytorch_tensor.dim() == 3:  # Si está en formato (H, W, C)
            pytorch_tensor = pytorch_tensor.permute(2, 0, 1)  # Convertir a (C, H, W)
        
        tensor_output.add_tensor(pytorch_tensor)

    g_prof.print_layer_times()
    print("Tipo de generated.data[0]:", type(tensor_output.data[0]))
    print("dimensiones [0]", tensor_output.data[0].dim())

    return tensor_output


    
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

def run_onnx_old(onnx_file, data_type, bs, inp_name):
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

def run_onnx(onnx_file, data_type, bs, inp_name):
    print("Archivo ONNX: " + onnx_file)
    import tensorrt as trt
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(G_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, G_LOGGER)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            print("Fallo al parsear el archivo ONNX.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None  # Devuelve None si hay error

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if data_type == trt.DataType.HALF:
        config.set_flag(trt.BuilderFlag.FP16)
    elif data_type == trt.DataType.INT8:
        config.set_flag(trt.BuilderFlag.INT8)

    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        print("Fallo al construir el motor serializado.")
        return None

    runtime = trt.Runtime(G_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        print("No se pudo deserializar el motor de ejecución.")
        return None

    # Asumiendo que time_inference devuelve el tensor o los tensores resultantes
    return time_inference(engine, bs, inp_name)
