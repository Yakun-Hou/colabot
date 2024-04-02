
import time
import pycuda.driver as cuda

import pycuda.autoinit
import numpy as np
import tensorrt as trt
import torch

trt_to_np_dtype = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT8: np.int8,
    trt.DataType.BOOL: np.bool,
    trt.DataType.INT32: np.int32,
}

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()
def allocate_buffers(engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            
            size = np.prod(engine.get_binding_shape(binding)) 
            dtype = trt_to_np_dtype[engine.get_binding_dtype(binding)]
            #print(binding+" size:"+str(size)+" dtype:"+str(dtype))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            #print(int(device_mem))
            bindings.append(int(device_mem))
            # Append ti the appropriate list
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream
class TensorRTTracker():
    def __init__(self,trt_file):
        self.trt_file=trt_file
        runtime = trt.Runtime(trt.Logger())
        with open(self.trt_file, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()
    def print_trt_file_name(self):
        print(self.trt_file)
    @staticmethod
    def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
        # Transfer input to the GPU
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference
        #print(bindings)
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs
        return [out.host for out in outputs]

    def track(self,input):
        #print(trt_file)
        
        
        
        #print(len(inputs))
        for i in range(len(input)):
            if isinstance(input[i],torch.Tensor):
                #print(input[i])
                self.inputs[i].host = input[i]  # np.random(1,3,1152,1152)  
                print(self.inputs[i].host.shape)
            else:
                self.inputs[i].host =input[i].astype(np.float32)
        # t = time.time()
        # t1 = time.time()
        num=0
        
        trt_outs = self.do_inference(self.context, self.bindings, self.inputs, self.outputs, self.stream)
        #print('do_infer time: ', (time.time() - t1))
        #self.print_trt_file_name()
        t1 = time.time()
        num=num+1
        #print(trt_outs[-1])
        #print(trt_outs[-2])
        return [trt_outs[-2],trt_outs[-1]]

if __name__=="__main__":

    filename="2.engine"
    np.random.seed(int(time.time()))
    template=np.ones((1,3,128,128))*0.5
    template_online=np.random.normal(0.5, 1, (1,3,128,128))
    search=np.random.normal(0.5, 1, (1,3,288,288))
    input=[template,template_online,search]
    tracker=TensorRTTracker(filename)
    