from General_functions import *

def Create_output(plasticity,zebrin,save_path,width,Noise_statemon_Coupled,PC_Statemon_Coupled,PC_Spikemon_Coupled, PC_rate_Coupled,DCN_Statemon_Coupled,DCN_Spikemon_Coupled,DCN_rate_Coupled,IO_Statemon_Coupled,IO_Spikemon_Coupled,IO_rate_Coupled,mon_N_PC_Coupled,Input_presyn_statemon_Coupled):
    time_start = time.monotonic()
    def time_checkpoint(name):
        nonlocal time_start
        e = time.monotonic()
        t = e - time_start
        time_start = e
        print(f'TIME: {name} took {t:.3f}s')
    time_checkpoint('starting save')

    Output_Noise_Coupled = {}
    for key in Noise_statemon_Coupled.recorded_variables.keys():
        Output_Noise_Coupled[key] = getattr(Noise_statemon_Coupled, key)


    Output_PC_Coupled = {}
    for key in PC_Statemon_Coupled.recorded_variables.keys():
        Output_PC_Coupled[key] = getattr(PC_Statemon_Coupled, key)
    Output_PC_Coupled['Spikemon'] = PC_Spikemon_Coupled.t/ms
    Output_PC_Spikes_Coupled = {}
    PC_Spikemon_Cells_Coupled = [[]]*PC_Spikemon_Coupled.values('t').__len__()
    for PC_spike in range(0,PC_Spikemon_Coupled.values('t').__len__()): 
        Output_PC_Spikes_Coupled[f'{PC_spike}'] = PC_Spikemon_Coupled.values('t')[PC_spike]
#     Output_PC_Coupled['Spikemon_Cells'] = PC_Spikemon_Cells_Coupled
    Output_PC_Coupled['Rate'] = PC_rate_Coupled.rate/Hz
    Output_PC_Coupled['Rate_time'] = PC_rate_Coupled.t/ms

    Output_DCN_Coupled = {}
    for key in DCN_Statemon_Coupled.recorded_variables.keys():
        Output_DCN_Coupled[key] = getattr(DCN_Statemon_Coupled, key)
    Output_DCN_Coupled['Spikemon'] = DCN_Spikemon_Coupled.t/ms
    Output_DCN_Spikes_Coupled = {}
    DCN_Spikemon_Cells_Coupled = [[]]*DCN_Spikemon_Coupled.values('t').__len__()
    for DCN_spike in range(DCN_Spikemon_Coupled.values('t').__len__()): 
        Output_DCN_Spikes_Coupled[f'{DCN_spike}'] = DCN_Spikemon_Coupled.values('t')[DCN_spike]
#     Output_DCN_Coupled['Spikemon_Cells'] = DCN_Spikemon_Cells_Coupled
    Output_DCN_Coupled['Rate'] = DCN_rate_Coupled.rate/Hz

    Output_IO_Coupled = {}
    for key in IO_Statemon_Coupled.recorded_variables.keys():
        Output_IO_Coupled[key] = getattr(IO_Statemon_Coupled, key)
    Output_IO_Coupled['Spikemon'] = IO_Spikemon_Coupled.t/ms
    Output_IO_Spikes_Coupled = {}
    IO_Spikemon_Cells_Coupled = [[]]*IO_Spikemon_Coupled.values('t').__len__()
    for IO_spike in range(IO_Spikemon_Coupled.values('t').__len__()): 
        Output_IO_Spikes_Coupled[f'{IO_spike}'] = IO_Spikemon_Coupled.values('t')[IO_spike]
#     Output_IO_Coupled['Spikemon_Cells'] = IO_Spikemon_Cells_Coupled
    Output_IO_Coupled['Rate'] = IO_rate_Coupled.rate/Hz

    if mon_N_PC_Coupled:
        Output_mon_N_PC_Coupled = {}
        for key in mon_N_PC_Coupled.recorded_variables.keys():
            Output_mon_N_PC_Coupled[key] = getattr(mon_N_PC_Coupled, key)

    Output_Input_presyn_Coupled = {}
    for key in Input_presyn_statemon_Coupled.recorded_variables.keys():
        Output_Input_presyn_Coupled[key] = getattr(Input_presyn_statemon_Coupled, key)


    sio.savemat(os.path.join(save_path, 'Output_Noise_Coupled.mat'), Output_Noise_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_PC_Coupled.mat'), Output_PC_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_PC_spikes_Coupled.mat'), Output_PC_Spikes_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_DCN_Coupled.mat'), Output_DCN_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_DCN_spikes_Coupled.mat'), Output_DCN_Spikes_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_IO_Coupled.mat'), Output_IO_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_IO_spikes_Coupled.mat'), Output_IO_Spikes_Coupled) 
    if mon_N_PC_Coupled:
        sio.savemat(os.path.join(save_path, 'Output_mon_N_PC_Coupled.mat'), Output_mon_N_PC_Coupled) 

    sio.savemat(os.path.join(save_path, 'Output_Input_presyn_Coupled.mat'), Output_Input_presyn_Coupled) 
    
   

def output_load_run(Cell_Name,name,seed_number,plasticity,zebrin,noise_gain,exp_run,net_name,path_data,parameters_value,f0):
    Output = {}
    tuning_range = {}
    net_path = path_data+'Frozen/Networks/'+net_name
    tun_path = path_data+'Simulations/Networks/'
    net_path_tun = tun_path+net_name
    seed_path_tun = net_path_tun+'/Seed_'+str(seed_number)
    run_path_tun = seed_path_tun+'/'+str(int(exp_run/msecond))+'ms'
    plasticity_path_tun = run_path_tun+'/'+plasticity
    zebrin_path_tun = plasticity_path_tun+'/Zebrin_'+zebrin
    noise_gain_path_tun = zebrin_path_tun + '/Noise_gain_'+str(int(noise_gain*10))
    if parameters_value['filtered']:
        if parameters_value["unfiltered"]:
            if f0 != 0:
                noise_gain_path_tun = noise_gain_path_tun +f'/Noise_filtered_{f0}'  
        else:
            noise_gain_path_tun = noise_gain_path_tun +f'/Noise_filtered_{f0}' 
    Output = sio.loadmat(noise_gain_path_tun+'/Output/Output_'+Cell_Name+name+'.mat', squeeze_me=True)
    return Output