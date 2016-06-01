import os
import queue, time, threading, datetime
import subprocess
from subprocess import DEVNULL
import itertools
import numpy as np

#%%
parameter_set = {
    'filename': 'conf.csv',
    'exe': 'term_project.exe',
    'image_dir': 'F:/VCSLab/Github/NonPhotorealisticRender/term_project/image/',
    'image': ['Lenna.png', 'motocycle.png'],
    ## bilateral, {windowSize}, {sigmaS}, {sigmaR}, {segment}
    'bilateral': {
        'windowSize': [21], 
        'sigmaS': [1.0, 3.0, 5.0], 
        'sigmaR': [2.0, 4.0, 6.0], 
        'segment': [21]},
    ## iteration, {quantize}, {edge}
    'iteration': {
        'quantize': [3],
        'edge': [3]},
    ## quantization, {bins}, {bottom}, {top}
    'quantization': {
        'bins': [3, 5, 7, 9], 
        'bottom': [0.0, 0.5, 1.0], 
        'top': [0.5, 1.0, 2.0, 4.0, 8.0]},
    ## edge detection (DoG),{windowSize}, {sigmaE}, {tau}, {phi}, {iteration}
    'DoG': {
        'windowSize': [21], 
        'sigmaE': [0.3, 0.5, 0.7], 
        'tau': [0.98], 
        'phi': [0.5, 1.0, 3.0, 5.0], 
        'iteration': [1, 3, 5]},
    ## image based warping (IBW), {windowSize}, {sigmaS}, {scale}
    'IBW': {
        'windowSize': [21], 
        'sigmaS': [0.5, 1.0, 1.5],
        'scale': [2.0, 3.0, 4.0]}
}

#%%
thread_num = 8

#%%
def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def gen_NPR_parameter(**parameter_set):
    for image in parameter_set['image']:
        for bilateral in dict_product(parameter_set['bilateral']):
            for iteration in dict_product(parameter_set['iteration']):
                for quantization in dict_product(parameter_set['quantization']):
                    if(quantization['bottom'] > quantization['top']): # skip when {bottom} larger than {top}
                        continue
                    for DoG in dict_product(parameter_set['DoG']):
                        for IBW in dict_product(parameter_set['IBW']):
                            yield {'filename': parameter_set['filename'], 'exe': parameter_set['exe'],
                                   'image_dir': parameter_set['image_dir'], 'image':image, 
                                   'bilateral': bilateral, 'iteration': iteration, 
                                   'quantization': quantization, 'DoG': DoG, 'IBW': IBW}

def write_config_file(**para):
    with open(para['filename'], 'w') as file:
        file.write('originalImage,{}{}\n'.format(para['image_dir'], para['image']))
        file.write('bilateral,{windowSize},{sigmaS},{sigmaR},{segment}\n'.format(**para['bilateral']))
        file.write('iteration,{quantize},{edge}\n'.format(**para['iteration']))
        file.write('quantization,{bins},{bottom},{top}\n'.format(**para['quantization']))
        file.write('DoG,{windowSize},{sigmaE},{tau},{phi},{iteration}\n'.format(**para['DoG']))
        file.write('IBW,{windowSize},{sigmaS},{scale}\n'.format(**para['IBW']))

def run_NPR(**para):
    write_config_file(**para)
    cmd = '{} {}'.format(para['exe'], para['filename'])
    subprocess.call(cmd)

def run_NPR_parameter_set(**parameter_set):
    for para in gen_NPR_parameter(**parameter_set):
        run_NPR(**para)

#%%
class Job:
    def __init__(self, image, bins, bottom, top):
        self.image = image
        self.bins = bins
        self.bottom = bottom
        self.top = top
    def do(self, parameter_set):
        ## set up working directory
        dirname = './p_{}_{}_{}_{}'.format(self.image, self.bins, self.bottom, self.top)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        ## copy exe file
        cmd = ['cp term_project.exe ', dirname]
        subprocess.call(''.join(cmd))
        
        ## replace parameter set
        kwargs = parameter_set.copy()
        # image
        kwargs['image'] = [self.image]
        # quantization
        kwargs['quantization']['bins'] = [self.bins]
        kwargs['quantization']['bottom'] = [self.bottom]
        kwargs['quantization']['top'] = [self.top]
        
        ## write to temp python file
        with open(dirname+'/tmp.py', 'w') as file:
            file.write('import sys\n')
            file.write('sys.path.append(r\'{}\')\n'.format(os.getcwd()))
            file.write('from run_thread import run_NPR_parameter_set\n')
            file.write('kwargs = {}\n'.format(kwargs))
            file.write('run_NPR_parameter_set(**kwargs)\n'.format(kwargs))

        ts = datetime.datetime.now()

        cmd = ['python tmp.py']
        subprocess.call(''.join(cmd), stdout=DEVNULL, cwd=dirname, shell=True)

        te = datetime.datetime.now() - ts
        print("\t[Info] Job({}, {}, {}) is done!".format(self.image, self.bins, self.bottom, self.top))
        print("\t[Info] Spending time={0}!".format(te))

def run_thread(parameter_set):
    que = queue.Queue()
    for image, bins, bottom, top in itertools.product(parameter_set['image'], parameter_set['quantization']['bins'], parameter_set['quantization']['bottom'], parameter_set['quantization']['top']):
        que.put(Job(image, bins, bottom, top))

    print("\t[Info] Queue size={0}...".format(que.qsize()))

    def doJob(*args):
        queue = args[0]
        while queue.qsize() > 0:
            job = queue.get()
            job.do(parameter_set)

    # Open threads
    thd = []
    for i in range(thread_num):
        thd.append(threading.Thread(target=doJob, name='Thd'+str(i), args=(que,)))

    # Start activity to digest queue.
    st = datetime.datetime.now()
    for t in thd:
        t.start()

    # Wait for all threads to terminate.
    while (threading.active_count() != 1):
        time.sleep(1)

    td = datetime.datetime.now() - st
    print("\t[Info] Spending time={0}!".format(td))

# %%
if __name__ == '__main__':
    run_thread(parameter_set)
    