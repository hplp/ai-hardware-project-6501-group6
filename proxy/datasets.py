import tqdm
import configparser
import pandas
from scalesim.scale_sim import scalesim

verbose = False
run_name = 'mm'
topology_file = 'mm.csv'
config_file = 'mm.cfg'
logpath = 'test_runs'
gemm_input = 'gemm'

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
type_of_dataflow = ['os', 'ws', 'is']


results = pandas.DataFrame(columns=['width', 'height', 'bufsize', 'dataflow', 'bandwidth', 'M', 'N', 'K', 'cycles'])
total_iterations = 4*4*4*3*4*4*4*4
with tqdm.tqdm(total=total_iterations, desc="Combined Progress") as pbar:
    for x in range(4):  # 8,16,32,64
        width = 2 ** (x+3)
        config['architecture_presets']['ArrayWidth'] = str(width)
        for y in range(4):  # 8,16,32,64
            height = 2 ** (y+3)
            config['architecture_presets']['ArrayHeight'] = str(height)
            for buf in range(4):    # 16,32,64,128 (kb)
                bufsize = 2 ** (buf + 4)
                config['architecture_presets']['IfmapSramSzkB'] = str(bufsize)
                config['architecture_presets']['FilterSramSzkB'] = str(bufsize)
                config['architecture_presets']['OfmapSramSzkB'] = str(bufsize)
                for dataflow in range(3):   # os, ws, is
                    config['architecture_presets']['Dataflow'] = type_of_dataflow[dataflow]
                    for bw in range(4):  # 8,16,24,32
                        bandwidth = (bw + 1) * 8
                        config['architecture_presets']['Bandwidth'] = str(bandwidth)
                        with open(config_file, 'w') as configfile:
                            config.write(configfile)
                            
                        # size of matrix multiplication
                        for i in range(4):   # 8,16,32,64
                            for j in range(4): # 8,16,32,64
                                for k in range(4): # 8,16,32,64
                                    M = 2 ** (i + 2)
                                    N = 2 ** (j + 2)
                                    K = 2 ** (k + 2)
                                    df = pandas.DataFrame({'Layer': [0], 'M': [M], 'N': [N], 'K': [K], '': ['']})
                                    df.to_csv(topology_file, index=False)
                                    # run sim
                                    s = scalesim(save_disk_space=True, verbose=verbose,
                                            config=config_file,
                                            topology=topology_file,
                                            input_type_gemm=gemm_input
                                            )
                                    s.run_scale(top_path=logpath)
                                    result_file = logpath+'/'+run_name+'/'+'COMPUTE_REPORT.csv'
                                    df = pandas.read_csv(result_file)
                                    cycles = df[' Total Cycles'].loc[0]
                                    # stalls = df[' Stall Cycles']
                                    # util = df[' Overall Util %']
                                    # compute_util = df[' Compute Util %']
                                    # new_row = pandas.DataFrame({'width': width, 'height': height, 
                                    #                         'bufsize': bufsize, 'dataflow': dataflow,
                                    #                         'bandwidth': bandwidth,
                                    #                         'M': M, 'N': N, 'K': K,
                                    #                         'cycles': cycles})
                                    results.loc[len(results)] = [width,height,bufsize,dataflow,bandwidth,M,N,K,cycles]
                                    
                                    pbar.update(1)
                                    if pbar.n % 5000 == 0:
                                        results.to_csv('dataset.csv', index=False)
results.to_csv('dataset.csv')
