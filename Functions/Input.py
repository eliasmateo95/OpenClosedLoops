from General_functions import *
from Neurons import *
from Synapses import *
from scipy.signal import butter, lfilter, filtfilt, sosfilt
import random


def Input_func(Parameters):
#     b2.device.reinit()
#     b2.device.activate()
#     b2.set_device('cpp_standalone', build_on_run=False)

    Noise = {"Noise_I": Noise_run(Parameters), "N_Cells_PF_events": Parameters["N_Cells_PF_events"]}
    for f0 in Parameters['f0_range']:
        Noise[f"filtered_noise_{f0}"] = filtered_noise(Parameters,f0,Noise["Noise_I"],Parameters["filter_order"])
    for simulation in Parameters['simulations']:
        noise_sim, IO_cop, iti_arr = sims(Parameters,simulation,Noise["Noise_I"])
        Noise[f"sim_{simulation}"] = noise_sim
        Noise[f"IO_copy_{simulation}"] = IO_cop
        Noise[f"iti_arr_{simulation}"] = iti_arr
          

    PC = {"PC_Values": PC_cell_values(Parameters['N_Cells_PC'],Parameters['PC_I_intrinsic']),
          "PC_variablity": Parameters['PC_variablity']}

    DCN = {"DCN_Values": DCN_cell_values(Parameters['N_Cells_DCN'],Parameters['DCN_I_intrinsic']),
           "DCN_variablity": Parameters['DCN_variablity'], "eqs_syn_IO_PC_pre": Parameters['eqs_syn_IO_PC_pre']}

    IO = {"IO_Values": IO_cell_values(Parameters['N_Cells_IO']),
          "IO_thresh": 'Vs>-30*mV',
          "eqs_IO_syn": ''' I_c_pre = (0.00125*mS/cm**2)*(0.6*e**(-((Vd_pre/mvolt-Vd_post/mvolt)/50)**2) + 0.4)*(Vd_pre-Vd_post) : metre**-2*amp (summed)'''}
#           "eqs_IO_syn": ''' I_c_pre = (0.00125*mS/cm**2)*(0.6*e**(-((Vd_pre/mvolt-Vd_post/mvolt)/50)**2) + 0.4)*(Vd_pre-Vd_post) : metre**-2*amp (summed)'''}


    Copy = {"rate_meas": 200*ms,
            "rate_meas_out": 200*ms,
            "rate_meas_PC": 100*ms,
            "rate_meas_out_PC": 100*ms,
            "tau_presyn": rand_params(10,ms,Parameters['N_Cells_PF'],(1.0/Parameters['N_Cells_PF'])),
            "tau_thresh_M": 10*ms,
            "eqs_syn_bcm_s_n_pc": '''
                              I_Noise_empty_post = weight_pre*I_pre : amp (summed)
                              I_Noise_post = (new_weight_pre)*(I_pre)*(1.0/N_Noise) : amp (summed)
        '''
    }

    Noise_PC_Synapse_Sources, Noise_PC_Synapse_Targets, Noise_PC_Synapse_Weights = Noise_PC_Weights(Parameters['N_Cells_PF'],Parameters['N_Cells_PC'])
    PC_DCN_Synapse_Sources, PC_DCN_Synapse_Targets = PC_DCN_Sources(Parameters['N_Cells_PC'],Parameters['N_Cells_DCN'],Parameters['N_PC_DCN_converge'],Parameters['N_PC_DCN_project'])
    DCN_IO_Synapse_Sources, DCN_IO_Synapse_Targets = DCN_IO_Sources(Parameters['N_Cells_PC'],Parameters['N_Cells_DCN'],Parameters['N_Cells_IO'])
    IO_PC_Synapse_Sources, IO_PC_Synapse_Targets = IO_PC_Sources(Parameters['N_Cells_IO'],Parameters['N_Cells_PC'],Parameters['IO_Conn_ratio'])
    IO_Synapse_Sources, IO_Synapse_Targets = IO_Sources(Parameters['N_Cells_IO'],Parameters['N_IO_project'])

    Synapses = {
        "Synapses": Parameters["Synapses"],
        "IO_Copy_Synapse_Targets":[x*Parameters['N_Cells_PF'] for x in IO_PC_Synapse_Sources],
        "IO_Copy_Synapse_Sources": np.repeat(IO_PC_Synapse_Sources,Parameters['N_Cells_PF']),
        "Noise_PC_Synapse_Sources": Noise_PC_Synapse_Sources,
        "Noise_PC_Synapse_Targets": Noise_PC_Synapse_Targets,
        "Noise_PC_Synapse_Weights": Noise_PC_Synapse_Weights,
        "PC_DCN_Synapse_Sources": PC_DCN_Synapse_Sources,
        "PC_DCN_Synapse_Targets": PC_DCN_Synapse_Targets,
        "DCN_IO_Synapse_Sources": DCN_IO_Synapse_Sources,
        "DCN_IO_Synapse_Targets": DCN_IO_Synapse_Targets,
        "IO_PC_Synapse_Sources": IO_PC_Synapse_Sources,
        "IO_PC_Synapse_Targets": IO_PC_Synapse_Targets,
        "IO_Synapse_Sources": IO_Synapse_Sources,
        "IO_Synapse_Targets": IO_Synapse_Targets
    }

    Initial={"Parameters":Parameters,"Noise":Noise,"PC":PC,"DCN":DCN, "IO":IO, "Copy":Copy, "Synapses":Synapses}
    Input = {}
    for names in Initial.keys():
        Input[names] = {}
        for key in Initial[names].keys():
            Input[names][key] = Initial[names][key]
            
    if Parameters["Record"]['Input'] == True:
        save_path_network = Parameters['path_data']+'Frozen/Networks/'+str(Parameters['N_Cells_PC'])+'PC_'+str(Parameters['N_Cells_DCN'])+'DCN_'+str(Parameters['N_Cells_IO'])+'IO'
        if os.path.exists(save_path_network) == False: 
                try:
                    os.mkdir(save_path_network)
                except OSError:
                    print ("Creation of the directory %s failed" % save_path_network)
                else:
                    print ("Successfully created the directory %s " % save_path_network)

        save_path_network_num = save_path_network+'/Seed_'+str(Parameters['net_num'])
        if os.path.exists(save_path_network_num) == False: 
                try:
                    os.mkdir(save_path_network_num)
                except OSError:
                    print ("Creation of the directory %s failed" % save_path_network_num)
                else:
                    print ("Successfully created the directory %s " % save_path_network_num)

        save_path_time = save_path_network_num+'/'+str(int(Parameters['exp_run']/msecond))+'ms'
        if os.path.exists(save_path_time) == False: 
                try:
                    os.mkdir(save_path_time)
                except OSError:
                    print ("Creation of the directory %s failed" % save_path_time)
                else:
                    print ("Successfully created the directory %s " % save_path_time) 
        

        Name = ""
        file_name = 'Frozen_'+str(int(Parameters['exp_run']/msecond))+'ms_'+str(Parameters['N_Cells_PC'])+'PC_'+str(Parameters['N_Cells_DCN'])+'DCN_'+str(Parameters['N_Cells_IO'])+'IO'+'_Seed_'+str(Parameters['net_num'])+'.mat'
        completeName = os.path.join(save_path_time, file_name)

        sio.savemat(completeName,Input) 
        print(f'Saved {Parameters["net_num"]}')
    b2.device.delete(code=True)
    return Input


def Read_Input(Frozen_data):
    class Noise:
        pass
    class Params:
        pass
    class Noise_frozen:
        pass
    class Values:
        pass
    class Synapses:
        pass
    Params.dt = dt = Frozen_data["Parameters"]["dt"]*second
    Params.dt_rec = Frozen_data["Parameters"]["dt_rec"]*second
    Params.tau_noise = Frozen_data["Parameters"]["tau_noise"]*second
    Params.exp_run = Frozen_data["Parameters"]["exp_run"]*second
    Params.width = Frozen_data["Parameters"]["width"]*second
    Params.N_Noise = Frozen_data["Parameters"]["N_Cells_PF"].item()
    Params.N_Cells_PC = Frozen_data["Parameters"]["N_Cells_PC"].item()
    Params.N_Cells_DCN = Frozen_data["Parameters"]["N_Cells_DCN"].item()
    Params.N_Cells_IO = Frozen_data["Parameters"]["N_Cells_IO"].item()
    Params.N_Copy = Frozen_data["Parameters"]["N_Copy"].item()
    Params.N_Copy_order = Frozen_data["Parameters"]["N_Copy_order"].item()
    Params.f0_range = Frozen_data["Parameters"]["f0_range"].item()
    ########################## Cell Values ############################
    Noise_frozen.Noise_I = Noise_I = Frozen_data["Noise"]["Noise_I"].item()
    Noise_frozen.N_Cells_PF_events = Frozen_data["Noise"]["N_Cells_PF_events"].item()
    Noise_frozen.Noise_filtered = {}
    f0_range = Params.f0_range
    if isinstance(f0_range,int):
        f0_range = [f0_range]
    for f0 in f0_range:
         Noise_frozen.Noise_filtered[f"filtered_noise_{f0}"] = Frozen_data["Noise"][f"filtered_noise_{f0}"].item()
    Noise_frozen.I_recorded = TimedArray(Noise_I.T, dt=dt)
    Noise_frozen.period = Params.exp_run
#     if hasattr(Frozen_data['Parameters'], 'simulations'):
    Params.simulations = Frozen_data["Parameters"]['simulations'].item()
    Noise_frozen.Noise_sim = {}
    simulations = Params.simulations
    if isinstance(simulations,str):
        simulations = [simulations]
    for simulation in simulations:
        simulation = simulation.strip()
        Noise_frozen.Noise_sim[f"sim_{simulation}"] = Frozen_data["Noise"][f"sim_{simulation}"].item()
        Noise_frozen.Noise_sim[f"IO_copy_{simulation}"] = Frozen_data["Noise"][f"IO_copy_{simulation}"].item()
        Noise_frozen.Noise_sim[f"iti_arr_{simulation}"] = Frozen_data["Noise"][f"iti_arr_{simulation}"].item()

    
    Values.PC_Values = Frozen_data["PC"]["PC_Values"].item()
    Values.PC_variablity = Frozen_data["PC"]["PC_variablity"].item()
    Values.DCN_Values = Frozen_data["DCN"]["DCN_Values"].item()
    Values.DCN_variablity = Frozen_data["DCN"]["DCN_variablity"].item()
    Values.IO_Values = Frozen_data["IO"]["IO_Values"].item()
    Values.IO_thresh = Frozen_data["IO"]["IO_thresh"].item()
    Values.eqs_IO_syn = Frozen_data["IO"]["eqs_IO_syn"].item()
    Values.rate_meas = Frozen_data["Copy"]["rate_meas"]*second
    Values.rate_meas_out = Frozen_data["Copy"]["rate_meas_out"]*second
    Values.rate_meas_PC = Frozen_data["Copy"]["rate_meas_PC"]*second
    Values.rate_meas_out_PC = Frozen_data["Copy"]["rate_meas_out_PC"]*second
    Values.tau_presyn = Frozen_data["Copy"]["tau_presyn"]*second
    Values.tau_thresh_M = Frozen_data["Copy"]["tau_thresh_M"]*second
    Values.eqs_syn_bcm_s_n_pc = Frozen_data["Copy"]["eqs_syn_bcm_s_n_pc"].item()
    Values.eqs_syn_IO_PC_pre = Frozen_data["DCN"]["eqs_syn_IO_PC_pre"].item()
    
#     for key, value in Frozen_data['IO'].items():
#         setattr(Values, key, value)
    ########################## Synapses ###############################
    Synapses.Synapses = Frozen_data["Synapses"]["Synapses"].item()
    Synapses.IO_Copy_Synapse_Targets = Frozen_data["Synapses"]["IO_Copy_Synapse_Targets"].item()
#     if "IO_Copy_Synapse_Sources" in Frozen_data["Synapses"]:
    Synapses.IO_Copy_Synapse_Sources = Frozen_data["Synapses"]["IO_Copy_Synapse_Sources"].item()
    Synapses.Noise_PC_Synapse_Sources = Frozen_data["Synapses"]["Noise_PC_Synapse_Sources"].item()
    Synapses.Noise_PC_Synapse_Targets = Frozen_data["Synapses"]["Noise_PC_Synapse_Targets"].item()  
    Synapses.Noise_PC_Synapse_Weights = Frozen_data["Synapses"]["Noise_PC_Synapse_Weights"].item() 
    Synapses.PC_DCN_Synapse_Sources = Frozen_data["Synapses"]["PC_DCN_Synapse_Sources"].item()
    Synapses.PC_DCN_Synapse_Targets = Frozen_data["Synapses"]["PC_DCN_Synapse_Targets"].item()
    Synapses.DCN_IO_Synapse_Sources = Frozen_data["Synapses"]["DCN_IO_Synapse_Sources"].item() 
    Synapses.DCN_IO_Synapse_Targets = Frozen_data["Synapses"]["DCN_IO_Synapse_Targets"].item() 
    Synapses.IO_PC_Synapse_Sources = Frozen_data["Synapses"]["IO_PC_Synapse_Sources"].item()
    Synapses.IO_PC_Synapse_Targets = Frozen_data["Synapses"]["IO_PC_Synapse_Targets"].item()
    Synapses.IO_Synapse_Sources = Frozen_data["Synapses"]["IO_Synapse_Sources"].item()
    Synapses.IO_Synapse_Targets = Frozen_data["Synapses"]["IO_Synapse_Targets"].item()
    
    return Params, Noise_frozen, Values, Synapses


def frozen_tun(seed_range,N_Cells_PC,N_Cells_DCN,N_Cells_IO,path_data,exp_run):
    Params = {}
    Noise_frozen = {}
    Values = {}
    Synaps = {}
    net_name = str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'
    net_path = path_data+'Frozen/Networks/'+net_name
    
    for seed_number in seed_range: 
        seed_path = net_path+'/Seed_'+str(seed_number)
        run_path = seed_path+'/'+str(int(exp_run/msecond))+'ms'
        frozen_path = run_path+'/Frozen_'+str(int(exp_run/msecond))+'ms_'+net_name+'_Seed_'+str(seed_number)+'.mat'
        ###################################################################
        ######################### Load Parameters #########################
        ###################################################################
        Frozen_data = sio.loadmat(frozen_path, squeeze_me=True)
        Params['Seed_'+str(seed_number)], Noise_frozen['Seed_'+str(seed_number)], Values['Seed_'+str(seed_number)], Synaps['Seed_'+str(seed_number)] = Read_Input(Frozen_data)

    return Params, Noise_frozen, Values, Synaps, net_name

def path_names(plasticity,tuning,path_data,exp_run,N_Cells_PC,N_Cells_DCN,N_Cells_IO,seed_number):
    net_path = path_data+'/Frozen/Networks/'+str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'
    seed_path = net_path+'/Seed_'+str(seed_number)
    run_path = seed_path+'/'+str(int(exp_run/msecond))+'ms'
    frozen_path = run_path+'/Frozen_'+str(int(exp_run/msecond))+'ms_'+str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'+'_Seed_'+str(seed_number)+'.mat'
    save_path_net = path_data+'/Tuning/Networks/'+str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'
    save_path_seed = save_path_net+'/Seed_'+str(seed_number)
    save_path_run = save_path_seed+'/'+str(int(exp_run/msecond))+'ms'
    save_path_plasticity = save_path_run+"/"+plasticity
    save_path_tuning = save_path_plasticity+'/'+tuning
    if os.path.exists(save_path_net) == False: os.mkdir(save_path_net)
    if os.path.exists(save_path_seed) == False: os.mkdir(save_path_seed)
    if os.path.exists(save_path_run) == False: os.mkdir(save_path_run)    
    if os.path.exists(save_path_plasticity) == False: os.mkdir(save_path_plasticity)    
    if os.path.exists(save_path_tuning) == False: os.mkdir(save_path_tuning)
        
    return net_path,seed_path,run_path,frozen_path,save_path_net,save_path_seed,save_path_run,save_path_plasticity,save_path_tuning

def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y


def sims(Parameters,simulation,Noise_I):
#     Parameters["N_Cells_PF_OU"]  = 5
    if simulation == 'eye_blink': 
        print(simulation)
        Parameters["N_Cells_PF_events"] = 2
        PF_Values = {}
        PF_Values["CS_pulse"] = [0]*(Parameters["N_Cells_PF_OU"])
        if Parameters["N_Cells_PF_events"] == 2: PF_Values["US_pulse"] = [0]*(Parameters["N_Cells_PF_OU"])
        if Parameters["N_Cells_PF_events"] >= 1:
            PF_Values["CS_pulse_duration"] = Parameters["CS_pulse_duration"]
            PF_Values["CS_pulse_amplitude"] = [Parameters["CS_pulse_amplitude"]]
            PF_Values["interpulse_duration"] = Parameters["interpulse_duration"]
            PF_Values["CS_pulse"][3] = PF_Values["CS_pulse_amplitude"][0]
        if Parameters["N_Cells_PF_events"] == 2:
            PF_Values["US_pulse"][3] = PF_Values["CS_pulse_amplitude"][0]
            PF_Values['US_pulse_initial'] = zeros(Parameters["N_Cells_PF_OU"])
            PF_Values['US_pulse_initial'][4] = Parameters["US_pulse_amplitude"]
            PF_Values['US_pulse_initial'][9] = Parameters["US_pulse_amplitude"]
            PF_Values["US_pulse_duration"] = Parameters["US_pulse_duration"]
            PF_Values["US_pulse_amplitude"] = [Parameters["US_pulse_amplitude"]]
            PF_Values["US_pulse"][4] = PF_Values["US_pulse_amplitude"][0]
            PF_Values["US_pulse"][9] = PF_Values["US_pulse_amplitude"][0]
            PF_Values['US_pulse'] = np.reshape(PF_Values['US_pulse'],[Parameters["N_Cells_PF_OU"],1]) 
            PF_Values['US_pulse_initial'] = np.reshape(PF_Values['US_pulse_initial'],[Parameters["N_Cells_PF_OU"],1]) 
        PF_Values['CS_pulse'] = np.reshape(PF_Values['CS_pulse'],[Parameters["N_Cells_PF_OU"],1])    
        I_pulse = zeros((Parameters["N_Cells_PF_OU"],int(Parameters["exp_run"]/Parameters['dt'])))
        transient = int(10*second/Parameters['dt'])
        iti_arr = [10]
        CS_pulse_dur = int(PF_Values['CS_pulse_duration']/Parameters['dt'])
        US_pulse_dur = int(PF_Values['US_pulse_duration']/Parameters['dt'])
        #Initial US pulse
        start = transient
        step = start + US_pulse_dur
        I_pulse[:,start:step] = PF_Values['US_pulse_initial']
        while step <= int(Parameters["exp_run"]/Parameters['dt']-CS_pulse_dur-transient):
            iti = random.randrange(8,12)
            #CS-US 
            start = step + int(iti*second/Parameters['dt'])
            step = start + CS_pulse_dur        
            I_pulse[:,start:step] = PF_Values['CS_pulse']
            start = step - int(PF_Values['US_pulse_duration']/Parameters['dt'])
            step = start + int(PF_Values['US_pulse_duration']/Parameters['dt']) 
            if step > int(Parameters["exp_run"]/Parameters['dt']-CS_pulse_dur-transient):
                break
            I_pulse[:,start:step] = PF_Values['US_pulse']  
            iti_arr.append(iti)
        #Final CS
        iti = random.randrange(8,13)
        iti_arr.append(iti)
        start = step + int(iti*second/Parameters['dt'])
        step = start + CS_pulse_dur        
        I_pulse[:,start:step] = PF_Values['CS_pulse']
        final_pulse = I_pulse[-1]
        noise_sim = (Noise_I/nA*amp+I_pulse)*nA
    elif simulation == 'pulse': 
        print(simulation)
        Parameters["N_Cells_PF_events"] = 2
        PF_Values = {}
        PF_Values["CS_pulse"] = [0]*(Parameters["N_Cells_PF_OU"])
        PF_Values["empty"] = [0]*(Parameters["N_Cells_PF_OU"])
        if Parameters["N_Cells_PF_events"] == 2: PF_Values["US_pulse"] = [0]*(Parameters["N_Cells_PF_OU"])
        if Parameters["N_Cells_PF_events"] >= 1:
            PF_Values["CS_pulse_duration"] = Parameters["CS_pulse_duration"]
            PF_Values["CS_pulse_amplitude"] = [Parameters["CS_pulse_amplitude"]]
            PF_Values["interpulse_duration"] = Parameters["interpulse_duration"]
            PF_Values["CS_pulse"][3] = PF_Values["CS_pulse_amplitude"][0]
        if Parameters["N_Cells_PF_events"] == 2:
        #     PF_Values["US_pulse"][3] = PF_Values["CS_pulse_amplitude"][0]
            PF_Values['US_pulse_initial'] = zeros(Parameters["N_Cells_PF_OU"])
            PF_Values['US_pulse_initial'][4] = Parameters["US_pulse_amplitude"]
            PF_Values['US_pulse_initial'][9] = Parameters["US_pulse_amplitude"]
            PF_Values["US_pulse_duration"] = Parameters["US_pulse_duration"]
            PF_Values["US_pulse_amplitude"] = [Parameters["US_pulse_amplitude"]]
            PF_Values["US_pulse"][4] = PF_Values["US_pulse_amplitude"][0]
            PF_Values["US_pulse"][9] = PF_Values["US_pulse_amplitude"][0]
            PF_Values['US_pulse'] = np.reshape(PF_Values['US_pulse'],[Parameters["N_Cells_PF_OU"],1]) 
            PF_Values['US_pulse_initial'] = np.reshape(PF_Values['US_pulse_initial'],[Parameters["N_Cells_PF_OU"],1]) 
        PF_Values['CS_pulse'] = np.reshape(PF_Values['CS_pulse'],[Parameters["N_Cells_PF_OU"],1])    
        PF_Values['empty'] = np.reshape(PF_Values['empty'],[Parameters["N_Cells_PF_OU"],1])   
        
        I_pulse = zeros((Parameters["N_Cells_PF_OU"],int(Parameters["exp_run"]/Parameters['dt'])))
        transient = int(10*second/Parameters['dt'])
        iti_arr = [10]
        CS_pulse_dur = int(3*ms/Parameters['dt'])
        US_CS_pulse_dur = int((PF_Values['CS_pulse_duration']-3*ms)/Parameters['dt'])
        US_pulse_dur = int(PF_Values['US_pulse_duration']/Parameters['dt'])
        #Initial US pulse
        start = transient
        step = start + US_pulse_dur
        I_pulse[:,start:step] = PF_Values['US_pulse_initial']
        while step <= int(Parameters["exp_run"]/Parameters['dt']-CS_pulse_dur-transient):
            iti = random.randrange(8,12)
            #CS-US 
            start = step + int(iti*second/Parameters['dt'])
            step = start + CS_pulse_dur  
            I_pulse[:,start:step] = PF_Values['CS_pulse']
            start = step
            step = start + US_CS_pulse_dur  
            I_pulse[:,start:step] = PF_Values["empty"]
            start = step - int(PF_Values['US_pulse_duration']/Parameters['dt'])
            step = start + int(PF_Values['US_pulse_duration']/Parameters['dt']) 
            if step > int(Parameters["exp_run"]/Parameters['dt']-CS_pulse_dur-transient):
                break
            I_pulse[:,start:step] = PF_Values['US_pulse']  
            iti_arr.append(iti)
        #Final CS
        iti = random.randrange(8,13)
        iti_arr.append(iti)
        start = step + int(iti*second/Parameters['dt'])
        step = start + CS_pulse_dur        
        I_pulse[:,start:step] = PF_Values['CS_pulse']
        final_pulse = I_pulse[-1]
        noise_sim = (Noise_I/nA*amp+I_pulse)*nA
    elif simulation == 'eye_blink_open': 
        print(simulation)
        Parameters["N_Cells_PF_events"] = 2
        PF_Values = {}
        PF_Values["CS_pulse"] = [0]*(Parameters["N_Cells_PF_OU"])
        if Parameters["N_Cells_PF_events"] == 2: PF_Values["US_pulse"] = [0]*(Parameters["N_Cells_PF_OU"])
        if Parameters["N_Cells_PF_events"] >= 1:
            PF_Values["CS_pulse_duration"] = Parameters["CS_pulse_duration"]
            PF_Values["CS_pulse_amplitude"] = [Parameters["CS_pulse_amplitude"]]
            PF_Values["interpulse_duration"] = Parameters["interpulse_duration"]
            PF_Values["CS_pulse"][3] = PF_Values["CS_pulse_amplitude"][0]
            PF_Values["CS_pulse"][8] = PF_Values["CS_pulse_amplitude"][0]
        if Parameters["N_Cells_PF_events"] == 2:
            PF_Values["US_pulse"][3] = PF_Values["CS_pulse_amplitude"][0]
            PF_Values["US_pulse"][8] = PF_Values["CS_pulse_amplitude"][0]
            PF_Values['US_pulse_initial'] = zeros(Parameters["N_Cells_PF_OU"])
            PF_Values['US_pulse_initial'][4] = Parameters["US_pulse_amplitude"]
            PF_Values['US_pulse_initial'][9] = Parameters["US_pulse_amplitude"]
            PF_Values["US_pulse_duration"] = Parameters["US_pulse_duration"]
            PF_Values["US_pulse_amplitude"] = [Parameters["US_pulse_amplitude"]]
            PF_Values["US_pulse"][4] = PF_Values["US_pulse_amplitude"][0]
            PF_Values["US_pulse"][9] = PF_Values["US_pulse_amplitude"][0]
            PF_Values['US_pulse'] = np.reshape(PF_Values['US_pulse'],[Parameters["N_Cells_PF_OU"],1]) 
            PF_Values['US_pulse_initial'] = np.reshape(PF_Values['US_pulse_initial'],[Parameters["N_Cells_PF_OU"],1]) 
        PF_Values['CS_pulse'] = np.reshape(PF_Values['CS_pulse'],[Parameters["N_Cells_PF_OU"],1])    
        I_pulse = zeros((Parameters["N_Cells_PF_OU"],int(Parameters["exp_run"]/Parameters['dt'])))
        transient = int(10*second/Parameters['dt'])
        iti_arr = [10]
        CS_pulse_dur = int(PF_Values['CS_pulse_duration']/Parameters['dt'])
        US_pulse_dur = int(PF_Values['US_pulse_duration']/Parameters['dt'])
        #Initial US pulse
        start = transient
        step = start + US_pulse_dur
        I_pulse[:,start:step] = PF_Values['US_pulse_initial']
        while step <= int(Parameters["exp_run"]/Parameters['dt']-CS_pulse_dur-transient):
            iti = random.randrange(8,12)
            #CS-US 
            start = step + int(iti*second/Parameters['dt'])
            step = start + CS_pulse_dur        
            I_pulse[:,start:step] = PF_Values['CS_pulse']
            start = step - int(PF_Values['US_pulse_duration']/Parameters['dt'])
            step = start + int(PF_Values['US_pulse_duration']/Parameters['dt']) 
            if step > int(Parameters["exp_run"]/Parameters['dt']-CS_pulse_dur-transient):
                break
            I_pulse[:,start:step] = PF_Values['US_pulse']  
            iti_arr.append(iti)
        #Final CS
        iti = random.randrange(8,13)
        iti_arr.append(iti)
        start = step + int(iti*second/Parameters['dt'])
        step = start + CS_pulse_dur        
        I_pulse[:,start:step] = PF_Values['CS_pulse']
        final_pulse = I_pulse[-1]
        noise_sim = (Noise_I/nA*amp+I_pulse)*nA
    elif simulation == 'pulse_open': 
        print(simulation)
        Parameters["N_Cells_PF_events"] = 2
        PF_Values = {}
        PF_Values["CS_pulse"] = [0]*(Parameters["N_Cells_PF_OU"])
        PF_Values["empty"] = [0]*(Parameters["N_Cells_PF_OU"])
        if Parameters["N_Cells_PF_events"] == 2: PF_Values["US_pulse"] = [0]*(Parameters["N_Cells_PF_OU"])
        if Parameters["N_Cells_PF_events"] >= 1:
            PF_Values["CS_pulse_duration"] = Parameters["CS_pulse_duration"]
            PF_Values["CS_pulse_amplitude"] = [Parameters["CS_pulse_amplitude"]]
            PF_Values["interpulse_duration"] = Parameters["interpulse_duration"]
            PF_Values["CS_pulse"][3] = PF_Values["CS_pulse_amplitude"][0]
            PF_Values["CS_pulse"][8] = PF_Values["CS_pulse_amplitude"][0]
        if Parameters["N_Cells_PF_events"] == 2:
        #     PF_Values["US_pulse"][3] = PF_Values["CS_pulse_amplitude"][0]
            PF_Values['US_pulse_initial'] = zeros(Parameters["N_Cells_PF_OU"])
            PF_Values['US_pulse_initial'][4] = Parameters["US_pulse_amplitude"]
            PF_Values['US_pulse_initial'][9] = Parameters["US_pulse_amplitude"]
            PF_Values["US_pulse_duration"] = Parameters["US_pulse_duration"]
            PF_Values["US_pulse_amplitude"] = [Parameters["US_pulse_amplitude"]]
            PF_Values["US_pulse"][4] = PF_Values["US_pulse_amplitude"][0]
            PF_Values["US_pulse"][9] = PF_Values["US_pulse_amplitude"][0]
            PF_Values['US_pulse'] = np.reshape(PF_Values['US_pulse'],[Parameters["N_Cells_PF_OU"],1]) 
            PF_Values['US_pulse_initial'] = np.reshape(PF_Values['US_pulse_initial'],[Parameters["N_Cells_PF_OU"],1]) 
        PF_Values['CS_pulse'] = np.reshape(PF_Values['CS_pulse'],[Parameters["N_Cells_PF_OU"],1])    
        PF_Values['empty'] = np.reshape(PF_Values['empty'],[Parameters["N_Cells_PF_OU"],1])   

        I_pulse = zeros((Parameters["N_Cells_PF_OU"],int(Parameters["exp_run"]/Parameters['dt'])))
        transient = int(10*second/Parameters['dt'])
        iti_arr = [10]
        CS_pulse_dur = int(3*ms/Parameters['dt'])
        US_CS_pulse_dur = int((PF_Values['CS_pulse_duration']-3*ms)/Parameters['dt'])
        US_pulse_dur = int(PF_Values['US_pulse_duration']/Parameters['dt'])
        #Initial US pulse
        start = transient
        step = start + US_pulse_dur
        I_pulse[:,start:step] = PF_Values['US_pulse_initial']
        while step <= int(Parameters["exp_run"]/Parameters['dt']-CS_pulse_dur-transient):
            iti = random.randrange(8,12)
            #CS-US 
            start = step + int(iti*second/Parameters['dt'])
            step = start + CS_pulse_dur  
            I_pulse[:,start:step] = PF_Values['CS_pulse']
            start = step
            step = start + US_CS_pulse_dur  
            I_pulse[:,start:step] = PF_Values["empty"]
            start = step - int(PF_Values['US_pulse_duration']/Parameters['dt'])
            step = start + int(PF_Values['US_pulse_duration']/Parameters['dt']) 
            if step > int(Parameters["exp_run"]/Parameters['dt']-CS_pulse_dur-transient):
                break
            I_pulse[:,start:step] = PF_Values['US_pulse']  
            iti_arr.append(iti)
        #Final CS
        iti = random.randrange(8,13)
        iti_arr.append(iti)
        start = step + int(iti*second/Parameters['dt'])
        step = start + CS_pulse_dur        
        I_pulse[:,start:step] = PF_Values['CS_pulse']
        final_pulse = I_pulse[-1]
        noise_sim = (Noise_I/nA*amp+I_pulse)*nA
    return noise_sim, final_pulse, iti_arr

            

def filtered_noise(Parameters,f0,Noise_I,order):
    if f0 == 1100:
        filtered_noise_arr = []
        for N_Noise in range(Parameters["N_Cells_PF"]):
            if N_Noise == 4:
                Fs = 1/(Parameters["dt"]/second)
                f = 2
                T = Parameters['exp_run']/second
                sample = T * Fs
                x = np.arange(sample)
                y = 0.25*np.sin(2 * np.pi * f * x / Fs) + 1.3
                y = y/amp
            else:
                y = Noise_I[N_Noise]/nA
            filtered_noise_arr.append(y*nA)
    elif f0 == 1200:   
        filtered_noise_arr = []
        for N_Noise,lowcut in enumerate(range(0,Parameters["N_Cells_PF"]*200,200)):
            if N_Noise == 4:
                Fs = 1/(Parameters["dt"]/second)
                f = 2
                T = Parameters['exp_run']/second
                sample = T * Fs
                x = np.arange(sample)
                y = 0.25*np.sin(2 * np.pi * f * x / Fs)+ 1.3
                y = y/amp
            else:
                fs = 1/(Parameters["dt"]/second)
                T = Parameters['exp_run']/second
                nsamples = T * fs
                t = np.arange(0, nsamples) / fs
                a = 0.02
                highcut = lowcut + 200
                if lowcut == 0: lowcut = 1
                x = Noise_I[N_Noise]/nA - Parameters["I_0_Noise"]/amp
                y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
                xf = rfftfreq(int(nsamples), 1 / fs)
                points_per_freq = len(xf) / (fs / 2)
                # Our target frequency is 4000 Hz
                target_idx = int(points_per_freq * 20)
                yf_no_Plasticity = rfft(y)
                yf_no_Plasticity[target_idx - 1 : target_idx + 2] = 0
                yf = (yf_no_Plasticity)
                yf_time = np.fft.irfft(yf)
                energ_x = sum(np.abs(x)**2)
                energ_yf = sum(np.abs(yf_time)**2)
                new_signal = Parameters["I_0_Noise"]/amp+yf_time*(energ_x**0.5)/(energ_yf**0.5)
                y = new_signal
            filtered_noise_arr.append(y*nA)
    elif f0 == 1300:   
        filtered_noise_arr = []
        for N_Noise in range(Parameters["N_Cells_PF"]):
            if N_Noise == 4:
                Fs = 1/(Parameters["dt"]/second)
                f = 2
                T = Parameters['exp_run']/second
                sample = T * Fs
                x = np.arange(sample)
                y = 0.25*np.sin(2 * np.pi * f * x / Fs) 
                y = Noise_I[N_Noise]/nA + y/amp
            else:
                y = Noise_I[N_Noise]/nA
            filtered_noise_arr.append(y*nA)
    elif f0 == 1400:   
        filtered_noise_arr = []
        for N_Noise in range(Parameters["N_Cells_PF"]):
            Fs = 1/(Parameters["dt"]/second)
            f = 2
            T = Parameters['exp_run']/second
            sample = T * Fs
            x = np.arange(sample)
            phase = 2 * np.pi*N_Noise/5
            y = 0.25*np.sin(2 * np.pi * f * x / Fs + phase) + 1.3
            y = y/amp
            filtered_noise_arr.append(y*nA)
    elif f0 == 1500:   
        filtered_noise_arr = []
        for N_Noise, f in enumerate([5, 11, 53, 101, 199]):
            Fs = 1/(Parameters["dt"]/second)
            T = Parameters['exp_run']/second
            sample = T * Fs
            x = np.arange(sample)
            y = 0.25*np.sin(2 * np.pi * f * x / Fs) + 1.3
            y = y/amp
            filtered_noise_arr.append(y*nA)
    else:
        # Plot the frequency response for a few different orders.
        fs = 1/(Parameters["dt"]/second)
        T = Parameters['exp_run']/second
        nsamples = T * fs
        t = np.arange(0, nsamples) / fs
        a = 0.02
        lowcut = f0
        step = 5
        if f0 >= 50:
            step = 25
        if f0 >= 100:
            step = 50
        if f0 == 800:
            lowcut = 1
            step = 800
        highcut = lowcut + step
        filtered_noise_arr = []
        if f0 == 1000: 
            for N_Noise,lowcut in enumerate(range(0,Parameters["N_Cells_PF"]*200,200)):
                highcut = lowcut + 200
                if lowcut == 0:
                    lowcut = 1
                if lowcut == Parameters["N_Cells_PF"]*200-200:
                    highcut = lowcut
                    lowcut = 1
                x = Noise_I[N_Noise]/nA - Parameters["I_0_Noise"]/amp
                y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
                xf = rfftfreq(int(nsamples), 1 / fs)
                points_per_freq = len(xf) / (fs / 2)
                # Our target frequency is 4000 Hz
                target_idx = int(points_per_freq * 20)
                yf_no_Plasticity = rfft(y)
                yf_no_Plasticity[target_idx - 1 : target_idx + 2] = 0
                yf = (yf_no_Plasticity)
                yf_time = np.fft.irfft(yf)
                energ_x = sum(np.abs(x)**2)
                energ_yf = sum(np.abs(yf_time)**2)
                new_signal = Parameters["I_0_Noise"]/amp+yf_time*(energ_x**0.5)/(energ_yf**0.5)
                filtered_noise_arr.append(new_signal*nA)
        else:
            for N_Noise in range(Parameters["N_Cells_PF"]):
                x = Noise_I[N_Noise]/nA - Parameters["I_0_Noise"]/amp
                y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
                xf = rfftfreq(int(nsamples), 1 / fs)
                points_per_freq = len(xf) / (fs / 2)
                # Our target frequency is 4000 Hz
                target_idx = int(points_per_freq * 20)
                yf_no_Plasticity = rfft(y)
                yf_no_Plasticity[target_idx - 1 : target_idx + 2] = 0
                yf = (yf_no_Plasticity)
                yf_time = np.fft.irfft(yf)
                energ_x = sum(np.abs(x)**2)
                energ_yf = sum(np.abs(yf_time)**2)
                new_signal = Parameters["I_0_Noise"]/amp+yf_time*(energ_x**0.5)/(energ_yf**0.5)
                filtered_noise_arr.append(new_signal*nA)
    return filtered_noise_arr