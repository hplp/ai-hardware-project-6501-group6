import os
import sys
import configparser
import pandas
import time
from scalesim.scale_sim import scalesim

run_name = 'test'
topology_file = 'test.csv'
config_file = 'test.cfg'
logpath = 'test_runs'
gemm_input = 'conv'
module = '../proxy/model_1_trained.pth'

pe_num = 64
bandwidth = 10
ibuf = 36
wbuf = 36
obuf = 36

config = configparser.ConfigParser()
config['general'] = {
    'run_name': run_name
}
config['architecture_presets'] = {
    'ArrayHeight':    16,
    'ArrayWidth':     16,
    'IfmapSramSzkB':    ibuf,
    'FilterSramSzkB':   wbuf,
    'OfmapSramSzkB':    obuf,
    'IfmapOffset':    0,
    'FilterOffset':   10000000,
    'OfmapOffset':    20000000,
    'Dataflow' : 'ws',
    'Bandwidth' : bandwidth,
    'MemoryBanks': 1
}
config['run_presets'] = {
    'InterfaceBandwidth': 'USER'
}

optimum_cycles = sys.maxsize
optimum_preset = {
    'ArrayHeight':    0,
    'ArrayWidth':     0,
    'IfmapSramSzkB':    ibuf,
    'FilterSramSzkB':   wbuf,
    'OfmapSramSzkB':    obuf,
    'IfmapOffset':    0,
    'FilterOffset':   10000000,
    'OfmapOffset':    20000000,
    'Dataflow' : 'X',
    'Bandwidth' : bandwidth,
    'MemoryBanks': 1
}

start = time.time()
for dataflow in ['ws', 'os', 'is']:
    for x in range(1, pe_num+1):
        if pe_num % x != 0:
            continue
        y = int(pe_num / x)
        config['architecture_presets']['ArrayWidth'] = str(x)
        config['architecture_presets']['ArrayHeight'] = str(y)
        config['architecture_presets']['Dataflow'] = dataflow
        
        with open('test.cfg', 'w') as configfile:
            config.write(configfile)

        s = scalesim(save_disk_space=True, verbose=True,
                    config=config_file,
                    topology=topology_file,
                    input_type_gemm=gemm_input
                    )
        s.run_scale(top_path=logpath)
        
        result_file = logpath+'/'+run_name+'/'+'COMPUTE_REPORT.csv'
        if not os.path.exists(result_file):
            print(f"ERROR: file not found :{result_file}")
            exit()
        
        df = pandas.read_csv(result_file)
        cycles = df[' Total Cycles']
        stalls = df[' Stall Cycles']
        util = df[' Overall Util %']
        compute_util = df[' Compute Util %']
        
        if cycles.loc[0] < optimum_cycles:
            optimum_cycles = cycles.loc[0]
            optimum_preset['ArrayWidth'] = x
            optimum_preset['ArrayHeight'] = y
            optimum_preset['Dataflow'] = dataflow
        
        
print("===========================END==============================")
print(optimum_cycles)
print(optimum_preset)

end = time.time()

print(f"Time cost: {end - start}s")