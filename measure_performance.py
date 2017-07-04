import time
import threading
import subprocess
import psutil
import pandas as pd

class MTPerform(threading.Thread):
    def __init__(self, gpu, interval=0.1):
        super(MTPerform, self).__init__()
        self.interval = interval
        self.gpu = gpu
        self.unit = 'gpu' if gpu >= 0 else 'cpu'
        
        self.attribute = ['memory.used','utilization.memory']
        self.df = []
        self.terminate = False
    
    def stop(self):
        self.terminate = True
    
    def reset(self):
        self.time = time.time()
        self.df = []
        
    def run(self):
        while True:
            self.measure(unit=self.unit)
            if self.terminate: break
            
    def write(self, fname):
        df = pd.DataFrame(self.df, columns=['memory','utilization'])
        df.to_csv(fname)
        dfm = df.mean()
        elapsed_time = time.time() - self.time
        print('elapsed time : {}'.format(elapsed_time))
        print('memory  (MB) : {}'.format(dfm['memory']))
        print('CPU util (%) : {}'.format(dfm['utilization']))
        
    def measure(self, unit='cpu'):
        if unit == 'cpu':
            (mem, use) = self.get_cpu_info()
        elif unit == 'gpu':
            (mem, use) = self.get_gpu_info()
        
        self.df.append([mem,use])
        time.sleep(self.interval)
    
    def get_gpu_info(self):
        cmd = 'nvidia-smi --id={} --query-gpu={} --format=csv,noheader,nounits'.format(
                self.gpu, ','.join(self.attribute))
        out = subprocess.check_output(cmd, shell=True)
        str_out = out.split(',')
        
        mem = int(str_out[0])
        use = int(str_out[1])
        
        return (mem, use)
        
    def get_cpu_info(self):
        mem = int(psutil.virtual_memory().used // 1e+6)
        use = int(psutil.cpu_percent())
        
        return (mem, use)
