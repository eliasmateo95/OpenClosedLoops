from General_functions import *


def rate_meas_func(rate_meas,PC,dt):
    syn = Synapses(PC, PC, '', on_pre={'add': 'recent_rate += 1/rate_meas', 
                                             'subtract': 'recent_rate -= 1/rate_meas'}, method='euler', dt=dt)     
    return syn

def rate_meas_PC_func(rate_meas_PC,PC,dt):
    syn = Synapses(PC, PC, '', on_pre={'add': 'recent_rate_100 += 1/rate_meas_PC', 
                                             'subtract': 'recent_rate_100 -= 1/rate_meas_PC'}, method='euler', dt=dt)     
    return syn



def presyn_inp(Input_presyn_record,I_recorded,plasticity,range_plasticity,N_Noise, dt, dt_rec):
    if plasticity in range_plasticity:
        eqs_presyn = '''
                rho_presyn = I_recorded(t,i)/nA*amp*Hz: Hz
                '''
        Input_presyn = NeuronGroup(N_Noise, eqs_presyn, threshold='True', method='euler', dt=dt)
        Input_presyn_statemon = StateMonitor(Input_presyn, variables=['rho_presyn'], record=Input_presyn_record, dt=dt_rec)

    else:
        eqs_presyn = '''
                rho_presyn = I_recorded(t,i)/nA*amp*Hz: Hz
                '''
        Input_presyn = NeuronGroup(N_Noise, eqs_presyn, threshold='True', method='euler', dt=dt)
        Input_presyn_statemon = StateMonitor(Input_presyn, variables=['rho_presyn'], record=Input_presyn_record, dt=dt_rec)
#         ampli = rand_params(30,Hz,N_Noise,(5/N_Noise))
#         for ii in range(0, N_Noise, 1):
#             Input_presyn.ampli[ii] = ampli[ii]
            
    return Input_presyn, Input_presyn_statemon


def conn_N_PC_func(conn_N_PC_record,plasticity,range_plasticity,N_Copy, Noise_PC_Synapse_Weights, dt, dt_rec):        
    ##### Hyperbolic      new_weight = clip((weight + 0.4*(delta_weight_BCM + delta_weight_CS)), 0,5) : 1 
    #####  new_weight = clip((weight +  0.4*(delta_weight_BCM + delta_weight_CS)), 0,5) : 1 
    ####    new_weight = clip((weight +  0.4*(delta_weight_BCM + delta_weight_CS)), 0,5) : 1 

    if plasticity in range_plasticity:
        eqs_Copy = '''
                I : amp  # copy of the noise current
                rho_PF : Hz 
                rho_PC : Hz
                weight : 1 (constant)
                new_weight = clip((weight +  0.4*(delta_weight_BCM + delta_weight_CS)), 0,5) : 1 
                ddelta_weight_CS/dt = (0-delta_weight_CS)/(350*msecond) : 1
                ddelta_weight_BCM/dt = 5*tanh(0.01*rho_PC*(rho_PC-thresh_M)/thresh_M*second)*rho_PF : 1 
                dphi_BCM/dt = tanh(0.01*rho_PC*(rho_PC-thresh_M)/thresh_M*second)*Hz : 1 
                phi = tanh(rho_PC*(rho_PC-thresh_M)/thresh_M*msecond)*Hz: Hz
                dthresh_M/dt = ((rho_PC**2) - thresh_M/tau_thresh_M) : Hz 
                tau_thresh_M : second
                slope : Hz
        '''
        conn_N_PC = NeuronGroup(N_Copy, eqs_Copy, method='euler',dt=dt)
        conn_N_PC.weight = Noise_PC_Synapse_Weights
#         mon_N_PC = StateMonitor(conn_N_PC , ['rho_PF','rho_PC','thresh_M','delta_weight_CS','new_weight','delta_weight_BCM'], record=conn_N_PC_record, dt=dt_rec) 
        mon_N_PC = StateMonitor(conn_N_PC , ['rho_PF','rho_PC','phi','phi_BCM','thresh_M','delta_weight_CS','new_weight','delta_weight_BCM','tau_thresh_M','I','slope'], record=conn_N_PC_record, dt=dt_rec) 
            
    else:    
        eqs_Copy = '''
                    I : amp  # copy of the noise current
                    rho_PF : Hz 
                    rho_PC : Hz
                    weight : 1 (constant)
                    new_weight = weight : 1 
                    delta_weight_CS : 1
                    delta_weight_BCM : 1 
                    phi : 1
        '''
        conn_N_PC = NeuronGroup(N_Copy, eqs_Copy, method='euler',dt=dt)
        conn_N_PC.weight = Noise_PC_Synapse_Weights
#         mon_N_PC = StateMonitor(conn_N_PC , ['rho_PF','rho_PC','delta_weight_CS','new_weight','delta_weight_BCM'], record=True, dt=dt_rec)
        mon_N_PC = StateMonitor(conn_N_PC , ['rho_PF','rho_PC','phi','delta_weight_CS','new_weight','delta_weight_BCM'], record=True, dt=dt_rec)
    return conn_N_PC, mon_N_PC


def IO_Sources(N_Cells_IO,N_IO_project):
#     IO_Synapse_Sources = np.concatenate([np.full(N_IO_project, IO_num) for IO_num in range(N_Cells_IO)])
#     IO_Synapse_Targets = np.concatenate([np.random.choice(np.setdiff1d(range(N_Cells_IO), [IO_num]), size=N_IO_project, replace=False) for IO_num in range(N_Cells_IO)])

    # Generate random connections
    IO_Synapse_Sources = np.concatenate([np.full(N_IO_project, IO_num) for IO_num in range(N_Cells_IO)])
    IO_Synapse_Targets = np.concatenate([np.random.choice(np.setdiff1d(range(N_Cells_IO), [IO_num]), size=N_IO_project, replace=False) for IO_num in range(N_Cells_IO)])

    # Ensure reciprocal connections (IOs receive from the IOs that are connected to them)
    reciprocal_sources = np.concatenate([IO_Synapse_Targets[i*N_IO_project:(i+1)*N_IO_project] for i in range(N_Cells_IO)])
    reciprocal_targets = np.concatenate([IO_Synapse_Sources[i*N_IO_project:(i+1)*N_IO_project] for i in range(N_Cells_IO)])

    IO_Synapse_Sources = np.concatenate([IO_Synapse_Sources, reciprocal_sources])
    IO_Synapse_Targets = np.concatenate([IO_Synapse_Targets, reciprocal_targets])

    return IO_Synapse_Sources, IO_Synapse_Targets


def Noise_PC_syn(Noise,PC,N_Noise,N_Cells_PC,dt,dt_rec):
    eqs_syn_Noise_PC_noSTDP = '''
        noise_weight : 1
        I_Noise_post = (noise_weight)*(I_pre)*(1.0/N_Noise) : amp (summed)
    '''
    Noise_PC_Synapse = Synapses(Noise, PC, eqs_syn_Noise_PC_noSTDP,dt=dt)    
    return Noise_PC_Synapse

# def Noise_PC_Weights(N_Noise,N_Cells_PC):
#     Noise_PC_Synapse_Sources = list(range(0,N_Noise))*N_Cells_PC
#     Noise_PC_Synapse_Targets = []
#     for pp in range(0,N_Cells_PC):
#         Noise_PC_Synapse_Targets += N_Noise * [pp]
#     Noise_PC_Synapse_Weights = []
#     for bb in range(0,N_Cells_PC):
#         w = np.random.random(5)+0.2
#         w = w/w.sum()*5
# #         w = [1.0]*5
#         Noise_PC_Synapse_Weights.extend(w)
#     Noise_PC_Synapse_Weights
    
#     return Noise_PC_Synapse_Sources, Noise_PC_Synapse_Targets, Noise_PC_Synapse_Weights 

def Noise_PC_Weights(N_Noise, N_Cells_PC):
    Noise_PC_Synapse_Sources = list(range(0, N_Noise)) * N_Cells_PC
    Noise_PC_Synapse_Targets = []
    for pp in range(0, N_Cells_PC):
        Noise_PC_Synapse_Targets += N_Noise * [pp]
    
    Noise_PC_Synapse_Weights = []
    
    for bb in range(0, N_Cells_PC):
        w = np.random.random(5) + 0.2  # Generate 5 weights
        w = w / w.sum() * 5  # Normalize and scale
        
        if bb < 100:  # For the first 100 PCs (closed-PCs)
            extended_w = list(w) + [0] * 5  # Non-zero weights for first 5 inputs, rest are zero
        else:  # For the last 100 PCs (open-PCs)
            extended_w = [0] * 5 + list(w)  # Non-zero weights for last 5 inputs, first 5 are zero
        
        Noise_PC_Synapse_Weights.extend(extended_w)
    
    return Noise_PC_Synapse_Sources, Noise_PC_Synapse_Targets, Noise_PC_Synapse_Weights

def PC_DCN_syn(PC,DCN,N_Cells_PC,N_Cells_DCN,eqs_pc_dcn,dt,dt_rec):
    PC_DCN_Synapse = Synapses(PC, DCN, on_pre=eqs_pc_dcn, delay=10*ms,dt=dt) 
#     PC_DCN_Synapse = Synapses(PC, DCN, on_pre='I_PC_post = 1*nA', delay=2*ms,dt=dt) 
    return PC_DCN_Synapse

def PC_DCN_Sources(N_Cells_PC,N_Cells_DCN,N_PC_DCN_converge,N_PC_DCN_project):
    if N_Cells_PC > 100: N_Cells_PC = 100        
    PC_DCN_Synapse_Sources = [PC_num for PC_num in range(N_Cells_PC) for _ in range(N_PC_DCN_project)]
    PC_DCN_Synapse_Targets = np.concatenate([np.random.choice(N_PC_DCN_converge, size=N_PC_DCN_project, replace=False) for _ in range(N_Cells_PC)])

    
    return PC_DCN_Synapse_Sources, PC_DCN_Synapse_Targets

# ,'g_c_post += -1*mS/cm**2/N_Cells_DCN'}
def DCN_IO_syn(DCN,IO,N_Cells_DCN,N_Cells_IO,w_IO_DCN,shunting,dt,dt_rec):
    if shunting == True:
        eqs_dcn_io = 'I_IO_DCN_post += w_IO_DCN/N_Cells_DCN*uA*cm**-2;g_c_post += -0.001/N_Cells_DCN*mS/cm**2'
    elif shunting == False:
        eqs_dcn_io = 'I_IO_DCN_post += w_IO_DCN/N_Cells_DCN*uA*cm**-2'
    IO_DCN_Synapse = Synapses(DCN,IO,on_pre = eqs_dcn_io, delay=50*ms, method = 'euler',dt=dt)
    return IO_DCN_Synapse

def DCN_IO_Sources(N_Cells_PC,N_Cells_DCN,N_Cells_IO):        
    # create an array of sources
    IO_DCN_Synapse_Sources = np.repeat(np.arange(N_Cells_DCN), 10)

    # create an array of targets
    IO_DCN_Synapse_Targets = np.zeros(N_Cells_DCN * 10, dtype=int)

    for i in range(N_Cells_DCN):
        # choose 10 unique targets for each source
        available_targets = np.setdiff1d(np.arange(N_Cells_IO), IO_DCN_Synapse_Targets[:i*10])
        if available_targets.size == 0:
            available_targets = np.arange(N_Cells_IO)
        IO_DCN_Synapse_Targets[i*10:(i+1)*10] = np.random.choice(available_targets, size=10, replace=False)
        
    return IO_DCN_Synapse_Sources, IO_DCN_Synapse_Targets


def IO_PC_syn(IO,PC,N_Cells_IO,N_Cells_PC,dt,dt_rec):
    IO_PC_Synapse = Synapses(IO, PC, on_pre ='w +=(1*nA)', delay=15*ms,method = 'euler',dt=dt)
    return IO_PC_Synapse

def IO_PC_Sources(N_Cells_IO,N_Cells_PC,IO_Conn_ratio):
    IO_PC_Synapse_Sources = []
    IO_PC_Synapse_Targets = []

    cell_range = range(N_Cells_PC)
    if N_Cells_PC > 100: cell_range = range(100)

    if int(N_Cells_IO/IO_Conn_ratio) >= cell_range[-1]+1:
        source_indices = np.random.choice(range(int(N_Cells_IO/IO_Conn_ratio)), cell_range[-1]+1, replace=False)
    else:
        source_indices = np.random.choice(range(N_Cells_IO), int(N_Cells_IO/IO_Conn_ratio), replace=False)

    for target_index in cell_range:
        if target_index < int(N_Cells_IO/IO_Conn_ratio):
            source_index = source_indices[target_index]
        else:
            source_index = np.random.choice(source_indices)
        IO_PC_Synapse_Sources.append(source_index)
        IO_PC_Synapse_Targets.append(target_index)
    if N_Cells_PC > 100: IO_PC_Synapse_Sources = IO_PC_Synapse_Sources*2  
    IO_PC_Synapse_Targets = range(200)
            
    return IO_PC_Synapse_Sources, IO_PC_Synapse_Targets



# def IO_PC_Sources(N_Cells_IO,N_Cells_PC,IO_Conn_ratio):
#     IO_PC_Synapse_Sources = []
#     IO_PC_Synapse_Targets = []
        
#     if int(N_Cells_IO/IO_Conn_ratio) >= N_Cells_PC:
#         source_indices = np.random.choice(range(int(N_Cells_IO/IO_Conn_ratio)), N_Cells_PC, replace=False)
#     else:
#         source_indices = np.random.choice(range(N_Cells_IO), int(N_Cells_IO/IO_Conn_ratio), replace=False)

#     for target_index in range(N_Cells_PC):
#         if target_index < int(N_Cells_IO/IO_Conn_ratio):
#             source_index = source_indices[target_index]
#         else:
#             source_index = np.random.choice(source_indices)
#         IO_PC_Synapse_Sources.append(source_index)
#         IO_PC_Synapse_Targets.append(target_index)
# #     IO_PC_Synapse_Targets = range(200)
            
#     return IO_PC_Synapse_Sources, IO_PC_Synapse_Targets