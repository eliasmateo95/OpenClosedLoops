from General_functions import *


def Noise_run(Parameters):
    PF_OU_Values = {}
    PF_OU_Values["I0"] = [Parameters["I_0_Noise"]*nA]*Parameters["N_Cells_PF_OU"] 
    PF_OU_Values["I_OU"] = [Parameters["I_0_Noise"]*nA]*Parameters["N_Cells_PF_OU"]
    PF_OU_Values["sigma"] = Parameters["sigma_arr"]*nA
#     PF_OU_Values["tau_noise"] = [Parameters['tau_noise']]*Parameters["N_Cells_PF_OU"]
    PF_OU_Values["tau_noise"] = Parameters["tau_noise_arr"]*ms
    PF_Values = {}
    PF_Values["US_pulse"] = [0]*Parameters["N_Cells_PF_OU"]
    PF_Values["no_pulse"] = [0]*Parameters["N_Cells_PF_OU"] 
    if Parameters["N_Cells_PF_events"] == 1:
        PF_Values["US_pulse_duration"] = Parameters["US_pulse_duration"]
        PF_Values["US_pulse_amplitude"] = [Parameters["US_pulse_amplitude"]+PF_OU_Values["I0"][0]/nA]
        PF_Values["interpulse_duration"] = Parameters["interpulse_duration"]
        PF_Values["US_pulse"].extend(PF_Values["US_pulse_amplitude"])
    if Parameters["N_Cells_PF_events"] == 2: 
        PF_Values["CS_pulse"] = [0]*Parameters["N_Cells_PF"]
        PF_Values["CS_pulse"][Parameters["N_Cells_PF_OU"]] = Parameters["CS_pulse_amplitude"]+PF_OU_Values["I0"][0]/nA     
        PF_Values["CS_pulse"][-1] = PF_OU_Values["I0"][0]/nA     
        PF_Values["US_pulse"].extend([Parameters["CS_pulse_amplitude"]+PF_OU_Values["I0"][0]/nA])
        PF_Values["US_pulse"].extend([Parameters["US_pulse_amplitude"]+PF_OU_Values["I0"][0]/nA])
        PF_Values["interpulse_duration"] = Parameters["interpulse_duration"]
        PF_Values["US_pulse_duration"] = Parameters["US_pulse_duration"]
        PF_Values["CS_pulse_duration"] = Parameters["CS_pulse_duration"]
    if Parameters["N_Cells_PF_events"] !=0:
        PF_Values["no_pulse"].extend([PF_OU_Values["I0"][0]/nA]*Parameters["N_Cells_PF_events"])
        PF_OU_Values["I0"].extend([0*nA]*Parameters["N_Cells_PF_events"])
        PF_OU_Values["I_OU"].extend([0*nA]*Parameters["N_Cells_PF_events"])
        PF_OU_Values["sigma"].extend([0*nA]*Parameters["N_Cells_PF_events"])
        PF_OU_Values["tau_noise"].extend([0.1*ms]*Parameters["N_Cells_PF_events"]) 

    eqs_PF = '''
        I = I_OU + pulse : amp
        dI_OU/dt = (I0 - I_OU)/tau_noise + sigma*xi*tau_noise**-0.5 : amp 
        I0 : amp
        weight : 1
        sigma : amp
        pulse : amp
        tau_noise : second
    '''

    PF = NeuronGroup(Parameters["N_Cells_PF"], eqs_PF, threshold = 'True', reset ='', method='euler',dt=Parameters["dt"])
    PF_statemon = StateMonitor(PF, variables=['I','I_OU','pulse','weight'], record=Parameters["Record"]["Noise"],dt=Parameters["dt"])
    for key in PF_OU_Values.keys():
        setattr(PF, key, PF_OU_Values[key])
        
        
 
    PF.pulse = PF_Values["no_pulse"]*nA 
    run(Parameters["exp_run"]) 
    
    b2.device.build(directory='output', compile=True, run=True, debug=False)
    
    return numpy.ascontiguousarray(PF_statemon.I, dtype=np.float64)



def Noise_neuron(Noise_record,N_Noise,I_recorded,Noise_Gain,exp_run,dt,dt_rec):
    eqs_noise = '''
    I = I_recorded(t,i)*amp : amp
    weight : 1 
    noise_gain : 1
    '''
    Noise = NeuronGroup(N_Noise, eqs_noise, threshold = 'True', reset ='', method='euler',dt=dt)
    Noise.noise_gain = Noise_Gain
    
    return Noise, StateMonitor(Noise, variables=['I'], record=Noise_record, dt=dt_rec)


def PC_cell_values(N_Cells_PC,PC_I_intrinsic):
    PC_Values = {}
    PC_Values["C"] = rand_params(75,1,N_Cells_PC,(1.0/N_Cells_PC))  #75*pF  #40 * pF  # 0.77*uF*cm**-2* #1090*pF
    PC_Values["gL"] = rand_params(30,1,N_Cells_PC,(1.0/N_Cells_PC))  #30 * nS
    PC_Values["EL"] = rand_params(-70.6,1,N_Cells_PC,(0.5/N_Cells_PC))  #-70.6 * mV
    PC_Values["VT"] = rand_params(-50.4,1,N_Cells_PC,(0.5/N_Cells_PC))  #-50.4 * mV
    PC_Values["DeltaT"] = rand_params(2,1,N_Cells_PC,(0.5/N_Cells_PC))  #2 * mV
    PC_Values["tauw"] = rand_params(144,1,N_Cells_PC,(2.0/N_Cells_PC))  #144*ms
    PC_Values["a"] = rand_params(4,1,N_Cells_PC,(0.5/N_Cells_PC))  #4*nS #2*PC_SingleNeuron.C[jj]/(144*ms) # 
    PC_Values["b"] = rand_params(0.0805,1,N_Cells_PC,(0.001/N_Cells_PC))  #0.0805*nA  #0*nA #
    PC_Values["Vr"] = rand_params(-70.6,1,N_Cells_PC,(0.5/N_Cells_PC))  #-70.6*mV
    PC_Values["v"] = rand_params(-70.6,1,N_Cells_PC,(0.5/N_Cells_PC))  #[-70.6*mV]*N_Cells_PC
    PC_Values["I_intrinsic"] = [PC_I_intrinsic*nA]*N_Cells_PC
    
    return PC_Values


def PC_neurons(PC_record,N_Cells_PC,PC_Values,dt,dt_rec):
    eqs_PC = PC_equations()
    PC = NeuronGroup(N_Cells_PC, model = eqs_PC, threshold='v>Vcut', reset="v=Vr; w+=b", method='euler', dt=dt)
#     for key in PC_Values.keys(): setattr(PC, key, PC_Values[key])
    for PC_num in range(N_Cells_PC):
        PC.C[PC_num] = PC_Values["C"].item()[PC_num]*pF
        PC.gL[PC_num] = PC_Values["gL"].item()[PC_num]*nS
        PC.EL[PC_num] = PC_Values["EL"].item()[PC_num]*mV
        PC.VT[PC_num] = PC_Values["VT"].item()[PC_num]*mV
        PC.DeltaT[PC_num] = PC_Values["DeltaT"].item()[PC_num]*mV
        PC.Vcut[PC_num] = PC.VT[PC_num] + 5*PC.DeltaT[PC_num]
        PC.tauw[PC_num] = PC_Values["tauw"].item()[PC_num]*ms
        PC.a[PC_num] = PC_Values["a"].item()[PC_num]*nS
        PC.b[PC_num] = PC_Values["b"].item()[PC_num]*nA
        PC.Vr[PC_num] = PC_Values["Vr"].item()[PC_num]*mV
        PC.I_Noise[PC_num] = 0.5*nA
        PC.v[PC_num] = PC_Values["v"].item()[PC_num]*mV
#         print(PC_Values["I_intrinsic"].item()[PC_num]*nA)
        PC.I_intrinsic[PC_num] = PC_Values["I_intrinsic"].item()[PC_num]*nA
    
#     PC_Statemon = StateMonitor(PC, variables = ['v','I_Noise'], record=PC_record, dt=dt_rec)
    PC_Statemon = StateMonitor(PC, variables = ['v','I_Noise'], record=PC_record, dt=dt_rec)
    PC_Spikemon = SpikeMonitor(PC)
    PC_rate = PopulationRateMonitor(PC)

    return PC, PC_Statemon, PC_Spikemon, PC_rate


def DCN_cell_values(N_Cells_DCN,DCN_I_intrinsic):
    DCN_Values = {}
    DCN_Values["C"] = rand_params(281,1,N_Cells_DCN,(1.0/N_Cells_DCN))  #281*pF  #40 * pF  # 0.77*uF*cm**-2* #1090*pF
    DCN_Values["gL"] = rand_params(30,1,N_Cells_DCN,(1.0/N_Cells_DCN))  #30 * nS
    DCN_Values["EL"] = rand_params(-70.6,1,N_Cells_DCN,(0.5/N_Cells_DCN))  #-70.6 * mV
    DCN_Values["VT"] = rand_params(-50.4,1,N_Cells_DCN,(0.5/N_Cells_DCN))  #-50.4 * mV
    DCN_Values["DeltaT"] = rand_params(2,1,N_Cells_DCN,(0.5/N_Cells_DCN))  #2 * mV
    DCN_Values["tauw"] = rand_params(30,1,N_Cells_DCN,(1.0/N_Cells_DCN))  #30*ms
    DCN_Values["a"] = rand_params(4,1,N_Cells_DCN,(0.5/N_Cells_DCN))  #4*nS #2*DCN_SingleNeuron.C[jj]/(144*ms) # 
    DCN_Values["b"] = rand_params(0.0805,1,N_Cells_DCN,(0.001/N_Cells_DCN))  #0.0805*nA  #0*nA #
    DCN_Values["Vr"] = rand_params(-65,1,N_Cells_DCN,(0.5/N_Cells_DCN))  #-65*mV
    DCN_Values["tauI"] = 30*1 #rand_params(30,ms,N_Cells_DCN,(1.0/N_Cells_DCN))  #30*ms
    DCN_Values["I_PC_max"] = [0*1]*N_Cells_DCN #rand_params(0.1,nA,N_Cells_DCN,(0.009/N_Cells_DCN))  #0*nA
    DCN_Values["v"] = rand_params(-70.6,1,N_Cells_DCN,(0.5/N_Cells_DCN))  #[-70.6*mV]*N_Cells_DCN
    DCN_Values["I_intrinsic"] = [DCN_I_intrinsic*1]*N_Cells_DCN #rand_params(DCN_I_intrinsic,nA,N_Cells_DCN,(DCN_variance/N_Cells_DCN))  #rand_params(2.5,nA,N_Cells_DCN,(0.001/N_Cells_DCN))  #[3*nA]*N_Cells_DCN   
    DCN_Values["I_PC"] = [0*1]*N_Cells_DCN  #rand_params(2.5,nA,N_Cells_DCN,(0.001/N_Cells_DCN))  #[3*nA]*N_Cells_DCN

    return DCN_Values

def DCN_neurons(DCN_record,N_Cells_DCN,DCN_Values,dt,dt_rec):
    eqs_DCN = DCN_equations()   
    DCN = NeuronGroup(N_Cells_DCN, model = eqs_DCN, threshold='v>Vcut', reset="v=Vr; w+=b", method='euler', dt=dt)
#     for key in DCN_Values.keys(): setattr(DCN, key, DCN_Values[key])
    for DCN_num in range(0,N_Cells_DCN,1):
        DCN.C[DCN_num] = DCN_Values["C"].item()[DCN_num]*pF
        DCN.gL[DCN_num] = DCN_Values["gL"].item()[DCN_num]*nS
        DCN.EL[DCN_num] = DCN_Values["EL"].item()[DCN_num]*mV
        DCN.VT[DCN_num] = DCN_Values["VT"].item()[DCN_num]*mV
        DCN.DeltaT[DCN_num] = DCN_Values["DeltaT"].item()[DCN_num]*mV
        DCN.Vcut[DCN_num] = DCN.VT[DCN_num] + 5*DCN.DeltaT[DCN_num]
        DCN.tauw[DCN_num] = DCN_Values["tauw"].item()[DCN_num]*ms
        DCN.a[DCN_num] = DCN_Values["a"].item()[DCN_num]*nS
        DCN.b[DCN_num] = DCN_Values["b"].item()[DCN_num]*nA
        DCN.Vr[DCN_num] = DCN_Values["Vr"].item()[DCN_num]*mV
        DCN.I_PC_max[DCN_num] = DCN_Values["I_PC_max"].item()[DCN_num]*nA
        DCN.v[DCN_num] = DCN_Values["v"].item()[DCN_num]*mV
        DCN.I_intrinsic[DCN_num] = DCN_Values["I_intrinsic"].item()[DCN_num]*nA
    DCN.I_PC = [0*nA]*N_Cells_DCN
    
    DCN_Statemon = StateMonitor(DCN, variables = ['v','I_PC'], record=DCN_record, dt=dt_rec)
    DCN_Spikemon = SpikeMonitor(DCN)
    DCN_rate = PopulationRateMonitor(DCN)

    return DCN, DCN_Statemon, DCN_Spikemon, DCN_rate



def IO_cell_values(N_Cells_IO):
    IO_Values = {}
    IO_Values["V_Na"] = rand_params(55,1 ,N_Cells_IO,(1.0/N_Cells_IO))  #55*mvolt
    IO_Values["V_K"] = rand_params(-75,1 ,N_Cells_IO,(1.0/N_Cells_IO))  #-75*mvolt
    IO_Values["V_Ca"] = rand_params(120,1 ,N_Cells_IO,(1.0/N_Cells_IO))  #120*mvolt
    IO_Values["V_l"] = rand_params(10,1 ,N_Cells_IO,(1.0/N_Cells_IO))  #10*mvolt 
    IO_Values["V_h"] = rand_params(-43,1 ,N_Cells_IO,(1.0/N_Cells_IO))  #-43*mvolt 
    IO_Values["Cm"] = rand_params(1,1 ,N_Cells_IO,(0.1/N_Cells_IO))  #1*uF*cm**-2 
    IO_Values["g_Na"] = rand_params(150,1,N_Cells_IO,(1.0/N_Cells_IO))  #150*mS/cm**2
    IO_Values["g_Kdr"] = rand_params(9.0,1,N_Cells_IO,(0.1/N_Cells_IO))  #9.0*mS/cm**2
    IO_Values["g_K_s"] = rand_params(5.0,1,N_Cells_IO,(0.1/N_Cells_IO))  #5.0*mS/cm**2
    IO_Values["g_h"] = rand_params(0.12,1,N_Cells_IO,(0.01/N_Cells_IO))  #0.12*mS/cm**2
    IO_Values["g_Ca_h"] = rand_params(4.5,1,N_Cells_IO,(0.1/N_Cells_IO))  #4.5*mS/cm**2
    IO_Values["g_K_Ca"] = rand_params(35,1,N_Cells_IO,(0.5/N_Cells_IO))  #35*mS/cm**2
    IO_Values["g_Na_a"] = rand_params(240,1,N_Cells_IO,(1.0/N_Cells_IO))  #240*mS/cm**2
    IO_Values["g_K_a"] = rand_params(240,1,N_Cells_IO,(0.5/N_Cells_IO))  #20*mS/cm**2
    IO_Values["g_ls"] = rand_params(0.016,1,N_Cells_IO,(0.001/N_Cells_IO))  #0.016*mS/cm**2
    IO_Values["g_ld"] = rand_params(0.015,1,N_Cells_IO,(0.001/N_Cells_IO))  #0.016*mS/cm**2
    IO_Values["g_la"] = rand_params(0.016,1,N_Cells_IO,(0.001/N_Cells_IO))  #0.016*mS/cm**2
    IO_Values["g_int"] = rand_params(0.13,1,N_Cells_IO,(0.001/N_Cells_IO))  #0.13*mS/cm**2
    IO_Values["p"] = rand_params(0.25,1,N_Cells_IO,(0.01/N_Cells_IO))  #0.25
    IO_Values["p2"] = rand_params(0.15,1,N_Cells_IO,(0.01/N_Cells_IO))   #0.15
    IO_Values["g_Ca_l"] =  rand_params(1.2,1,N_Cells_IO,(0.01/N_Cells_IO))   #[0.75*1]*N_Cells_IO

    return IO_Values

def IO_neurons(IO_record,N_Cells_IO,Noise_frozen,IO_Values,thresh,I_recorded_copy,shunting,f0,dt,dt_rec):
    eqs_IO = IO_equations(Noise_frozen,I_recorded_copy,shunting,f0,dt)
    IO = NeuronGroup(N_Cells_IO, model = eqs_IO, threshold=thresh,refractory=thresh, method = 'euler',dt=dt) 
    for IO_num in range(0, N_Cells_IO, 1):
        IO.V_Na[IO_num] = IO_Values["V_Na"].item()[IO_num]*mV
        IO.V_K[IO_num] = IO_Values["V_K"].item()[IO_num]*mV
        IO.V_Ca[IO_num] = IO_Values["V_Ca"].item()[IO_num]*mV
        IO.V_l[IO_num] = IO_Values["V_l"].item()[IO_num]*mV
        IO.V_h[IO_num] = IO_Values["V_h"].item()[IO_num]*mV
        IO.Cm[IO_num] = IO_Values["Cm"].item()[IO_num]*uF*cm**-2
        IO.g_Na[IO_num] = IO_Values["g_Na"].item()[IO_num]*mS/cm**2
        IO.g_Kdr[IO_num] = IO_Values["g_Kdr"].item()[IO_num]*mS/cm**2
        IO.g_K_s[IO_num] = IO_Values["g_K_s"].item()[IO_num]*mS/cm**2
        IO.g_h[IO_num] = IO_Values["g_h"].item()[IO_num]*mS/cm**2
        IO.g_Ca_h[IO_num] = IO_Values["g_Ca_h"].item()[IO_num]*mS/cm**2
        IO.g_K_Ca[IO_num] = IO_Values["g_K_Ca"].item()[IO_num]*mS/cm**2
        IO.g_Na_a[IO_num] = IO_Values["g_Na_a"].item()[IO_num]*mS/cm**2
        IO.g_K_a[IO_num] = IO_Values["g_K_a"].item()[IO_num]*mS/cm**2
        IO.g_ls[IO_num] = IO_Values["g_ls"].item()[IO_num]*mS/cm**2
        IO.g_ld[IO_num] = IO_Values["g_ld"].item()[IO_num]*mS/cm**2
        IO.g_la[IO_num] = IO_Values["g_la"].item()[IO_num]*mS/cm**2
        IO.g_int[IO_num] = IO_Values["g_int"].item()[IO_num]*mS/cm**2
        IO.p[IO_num] = IO_Values["p"].item()[IO_num]
        IO.p2[IO_num] = IO_Values["p2"].item()[IO_num]
        IO.g_Ca_l[IO_num] =  IO_Values["g_Ca_l"].item()[IO_num]*mS/cm**2
        
        record_arr = ['Vs','I_IO_DCN','I_OU_Copy']
        IO_Statemon = StateMonitor(IO, variables = record_arr, record = IO_record, dt=dt_rec)
#         IO_Statemon = StateMonitor(IO, variables = ['Vs','I_IO_DCN','g_c','I_c'], record = IO_record, dt=dt_rec)
        IO_Spikemon = SpikeMonitor(IO, variables=['Vs'])
        IO_rate = PopulationRateMonitor(IO)
        
    return IO, IO_Statemon, IO_Spikemon, IO_rate

                                         
                                         
def IO_coup_syn(IO,eqs_IO_syn):
    IO_synapse = Synapses(IO, IO, eqs_IO_syn)
    return IO_synapse
                
                
                
def PC_equations():
    eqs_PC = """
    dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + I_Noise + I_intrinsic  -w)/C : volt
    dw/dt = (a*(v - EL) - w)/(tauw) : amp

    I_intrinsic : amp
    I_Noise  : amp  
    I_Noise_empty : amp
    
    C : farad
    gL : siemens 
    EL : volt
    VT : volt
    DeltaT : volt
    Vcut : volt
    tauw : second
    a : siemens
    b : ampere
    Vr : volt
    
    New_recent_rate = recent_rate : Hz
    PC_short_rate = recent_rate_100 : Hz
    recent_rate : Hz
    recent_rate_100 : Hz
    dtry_new_bcm/dt = -try_new_bcm/(50*ms) : Hz
    """
    return eqs_PC  

def DCN_equations():
    eqs_DCN = """
    dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + I_intrinsic - I_PC - w)/C : volt
    dw/dt = (a*(v - EL) - w)/tauw : amp

    dI_PC/dt = (I_PC_max-I_PC)/(30*ms) : amp

    I_intrinsic : amp

    C : farad
    gL : siemens 
    EL : volt
    taum : second
    VT : volt
    DeltaT : volt
    Vcut : volt
    tauw : second
    a : siemens
    b : ampere
    Vr : volt
    tauI : second
    I_PC_max : amp
    """
    return eqs_DCN

              
# 0.00125                
def IO_equations(Noise_frozen,I_recorded_copy,shunting,f0,dt):
    eqs_IO_V = '''
    dVs/dt = (-(I_ds + I_ls + I_Na + I_Ca_l + I_K_dr + I_K_s + I_h + I_as) + Iapp_s + I_OU + I_IO_DCN)/Cm      : volt    
    dVd/dt = (-(I_sd + I_ld + I_Ca_h + I_K_Ca + I_c) + Iapp_d)/Cm                               : volt
    dVa/dt = (-(I_K_a + I_sa + I_la + I_Na_a))/Cm                                               : volt
    dI_IO_DCN/dt = (0*uA*cm**-2 - I_IO_DCN)/(30*ms)                                             : amp*meter**-2
    I_c                                                                                         : metre**-2*amp
    Iapp_s                                                                                      : metre**-2*amp
    Iapp_d                                                                                      : metre**-2*amp
    I_OU_Copy                                                                                   : metre**-2*amp
    '''
    if f0 == 'eye_blink' or f0 == 'pulse' or f0 == 'eye_blink_open' or f0 == 'pulse_open':
        print(f'{f0} IO_OU_Copy')
        eqs_IO_V_copy = '''
        dVs/dt = (-(I_ds + I_ls + I_Na + I_Ca_l + I_K_dr + I_K_s + I_h + I_as) + Iapp_s + I_OU + I_IO_DCN + I_OU_Copy )/Cm      : volt
        dVd/dt = (-(I_sd + I_ld + I_Ca_h + I_K_Ca + I_c) + Iapp_d)/Cm                               : volt
        dVa/dt = (-(I_K_a + I_sa + I_la + I_Na_a))/Cm                                               : volt
        dI_IO_DCN/dt = (0*uA*cm**-2 - I_IO_DCN)/(30*ms)                                             : amp*meter**-2
        I_c                                                                                         : metre**-2*amp
        Iapp_s                                                                                      : metre**-2*amp
        Iapp_d                                                                                      : metre**-2*amp
        I_OU_Copy = (I_recorded_copy(t-4*ms))*uA*cm**-2                                             : metre**-2*amp
#         '''
#     if Noise_frozen.N_Cells_PF_events == 2:
#         I_recorded = Noise_frozen.I_recorded
#         eqs_IO_V_copy = '''
#         dVs/dt = (-(I_ds + I_ls + I_Na + I_Ca_l + I_K_dr + I_K_s + I_h + I_as) + Iapp_s + I_OU + I_IO_DCN)/Cm      : volt
#         dVd/dt = (-(I_sd + I_ld + I_Ca_h + I_K_Ca + I_c) + Iapp_d)/Cm                               : volt
#         dVa/dt = (-(I_K_a + I_sa + I_la + I_Na_a) + I_OU_Copy )/Cm                                  : volt
#         dI_IO_DCN/dt = (0*uA*cm**-2 - I_IO_DCN)/(30*ms)                                             : amp*meter**-2
#         I_c                                                                                         : metre**-2*amp
#         Iapp_s                                                                                      : metre**-2*amp
#         Iapp_d                                                                                      : metre**-2*amp
#         I_OU_Copy = 3*(I_recorded(t-4*ms,4)*10**9-1.3)*uA*cm**-2                                    : metre**-2*amp
# #         '''
    eqs_IO_Ca = '''
    dCa/dt = (-3*I_Ca_h*((uamp / cm**2)**-1)*mM - 0.075*Ca)/ms                                  : mM
    '''
    eqs_IO_Isom = '''
    I_as    = (g_int/(1-p2))*(Vs-Va)                                                            : metre**-2*amp
    I_ls    = g_ls*(Vs-V_l)                                                                     : metre**-2*amp
    I_ds    = (g_int/p)*(Vs-Vd)                                                                 : metre**-2*amp
    I_Na    = g_Na*m_inf**3*h*(Vs-V_Na)                                                         : metre**-2*amp
    I_Ca_l  = g_Ca_l*k*k*k*l*(Vs-V_Ca)                                                          : metre**-2*amp
    I_K_dr  = g_Kdr*n*n*n*n*(Vs-V_K)                                                            : metre**-2*amp
    I_h     = g_h*q*(Vs-V_h)                                                                    : metre**-2*amp
    I_K_s   = g_K_s*(x_s**4)*(Vs-V_K)                                                           : metre**-2*amp
    '''
    eqs_IO_Iden = '''
    I_sd    = (g_int/(1-p))*(Vd-Vs)                                                             : metre**-2*amp
    I_ld    = g_ld*(Vd-V_l)                                                                     : metre**-2*amp
    I_Ca_h  = g_Ca_h*r*r*(Vd-V_Ca)                                                              : metre**-2*amp
    I_K_Ca  = g_K_Ca*s*(Vd-V_K)                                                                 : metre**-2*amp
    '''
    eqs_IO_Iax = '''
    I_K_a  = g_K_a *x_a**4*(Va-V_K)                                                             : metre**-2*amp
    I_sa   = (g_int/p2)*(Va-Vs)                                                                 : metre**-2*amp
    I_la   = g_la*(Va-V_l)                                                                      : metre**-2*amp
    I_Na_a = g_Na_a*m_a**3*h_a*(Va-V_Na)                                                        : metre**-2*amp
    '''
    eqs_IO_activation = '''
    dh/dt = (h_inf - h)/tau_h                                                                   : 1
    dk/dt = (k_inf - k)/tau_k                                                                   : 1
    dl/dt = (l_inf - l)/tau_l                                                                   : 1
    dn/dt = (n_inf - n)/tau_n                                                                   : 1
    dq/dt = (q_inf - q)/tau_q                                                                   : 1
    dr/dt = (r_inf - r)/tau_r                                                                   : 1
    ds/dt = (s_inf - s)/tau_s                                                                   : 1
    m_a = m_inf_a                                                                               : 1
    dh_a/dt = (h_inf_a - h_a)/tau_h_a                                                           : 1
    dx_a/dt = (x_inf_a - x_a)/tau_x_a                                                           : 1
    dx_s/dt = (x_inf_s - x_s)/tau_x_s                                                           : 1
    '''
    eqs_IO_inf = '''
    m_inf   = 1/(1+e**(-(Vs/mvolt+30)/5.5))                                                     : 1 (constant over dt)
    h_inf   = 1/(1+e**((Vs/mvolt+70)/5.8))                                                      : 1 (constant over dt)
    k_inf   = 1/(1+e**(-(Vs/mvolt+61.0)/4.2))                                                   : 1 (constant over dt)
    l_inf   = 1/(1+e**((Vs/mvolt+85.5)/8.5))                                                    : 1 (constant over dt)
    n_inf   = 1/(1+e**(-(Vs/mvolt+3)/10))                                                       : 1 (constant over dt)
    q_inf   = 1/(1+e**((Vs/mvolt+80.0)/(4.0)))                                                  : 1 (constant over dt)
    r_inf   = alpha_r/(alpha_r + beta_r)                                                        : 1 (constant over dt)
    s_inf   = alpha_s/(alpha_s+beta_s)                                                          : 1 (constant over dt)
    m_inf_a = 1/(1+(e**((-30.0-Va/mvolt)/ 5.5)))                                                : 1 (constant over dt)
    h_inf_a = 1/(1+(e**((-60.0-Va/mvolt)/-5.8)))                                                : 1 (constant over dt)
    x_inf_a = alpha_x_a/(alpha_x_a+beta_x_a)                                                    : 1 (constant over dt)
    x_inf_s = alpha_x_s/(alpha_x_s + beta_x_s)                                                  : 1 (constant over dt)
    '''
    eqs_IO_tau = '''
    tau_h   = 3*msecond*e**(-(Vs/mvolt+40)/33)                                                  : second (constant over dt)
    tau_k   = 5.0*msecond                                                                       : second (constant over dt)
    tau_l   = 1.0*msecond*(35.0+(20.0*exp((Vs/mvolt+160.0)/30.0))/(1+exp((Vs/mvolt+84.0)/7.3))) : second (constant over dt)
    tau_n   = 5.0*msecond + (47*e**((Vs/mvolt + 50)/900))*msecond                               : second (constant over dt)
    tau_q   = 1.0*msecond/(e**((-0.086*Vs/mvolt-14.6))+e**((0.07*Vs/mvolt-1.87)))               : second (constant over dt)
    tau_r   = 5.0*msecond/(alpha_r + beta_r)                                                    : second (constant over dt)
    tau_s   = 1.0*msecond/(alpha_s + beta_s)                                                    : second (constant over dt)
    tau_h_a = 1.5*msecond*e**((-40.0-Va/mvolt)/33.0)                                            : second (constant over dt)
    tau_x_a = 1.0*msecond/(alpha_x_a + beta_x_a)                                                : second (constant over dt)
    tau_x_s = 1.0*msecond/(alpha_x_s + beta_x_s)                                                : second (constant over dt)
    '''
    eqs_IO_alpha = '''
    alpha_m   = (0.1*(Vs/mvolt + 41.0))/(1-e**(-(Vs/mvolt+41.0)/10.0))                          : 1 (constant over dt)
    alpha_h   = 5.0*e**(-(Vs/mvolt+60.0)/15.0)                                                  : 1 (constant over dt)
    alpha_n   = (Vs/mvolt + 41.0)/(1-e**(-(Vs/mvolt+41.0)/10.0))                                : 1 (constant over dt)
    alpha_r   = 1.7/(1+e**(-(Vd/mvolt - 5.0)/13.9))                                             : 1 (constant over dt)
    alpha_s   = ((0.00002*Ca/mM)*int((0.00002*Ca/mM)<0.01) + 0.01*int((0.00002*Ca/mM)>=0.01))   : 1 (constant over dt)
    alpha_x_a = 0.13*(Va/mvolt + 25.0)/(1-e**(-(Va/mvolt+25.0)/10.0))                           : 1 (constant over dt)
    alpha_x_s = 0.13*(Vs/mvolt + 25.0)/(1-e**(-(Vs/mvolt+25.0)/10.0))                           : 1 (constant over dt)
    '''

    eqs_IO_beta = '''
    beta_m = 9.0*e**(-(Vs/mvolt+60.0)/20.0)                                                     : 1 (constant over dt)
    beta_h = (Vs/mvolt+50.0)/(1-e**(-(Vs/mvolt+50.0)/10.0))                                     : 1 (constant over dt)
    beta_n = 12.5*e**(-(Vs/mvolt+51.0)/80.0)                                                    : 1 (constant over dt)
    beta_r = 0.02*(Vd/mvolt + 8.5)/(e**((Vd/mvolt + 8.5)/5.0)-1)                                : 1 (constant over dt)
    beta_s = 0.015                                                                              : 1 (constant over dt)
    beta_x_a  = 1.69*e**(-0.0125*(Va/mvolt + 35.0))                                             : 1 (constant over dt)
    beta_x_s  = 1.69*e**(-0.0125*(Vs/mvolt+ 35.0))                                              : 1 (constant over dt)
    '''

    eqs_vector = '''
    V_Na                                                                                        : volt
    V_K                                                                                         : volt
    V_Ca                                                                                        : volt
    V_l                                                                                         : volt
    V_h                                                                                         : volt
    Cm                                                                                          : farad*meter**-2
    g_Na                                                                                        : siemens/meter**2
    g_Kdr                                                                                       : siemens/meter**2
    g_Ca_l                                                                                      : siemens/meter**2
    g_h                                                                                         : siemens/meter**2
    g_Ca_h                                                                                      : siemens/meter**2
    g_K_Ca                                                                                      : siemens/meter**2
    g_ls                                                                                        : siemens/meter**2
    g_ld                                                                                        : siemens/meter**2
    g_int                                                                                       : siemens/meter**2
    g_Na_a                                                                                      : siemens/meter**2
    g_K_a                                                                                       : siemens/meter**2
    g_la                                                                                        : siemens/meter**2
    g_K_s                                                                                       : siemens/meter**2
    p                                                                                           : 1
    p2                                                                                          : 1
    '''

    eqs_noise = """
    dI_OU/dt = (I0_OU - I_OU)/tau_noise + sigma_OU*xi*tau_noise**-0.5                           : amp*meter**-2 
    I0_OU                                                                                       : amp*meter**-2 
    sigma_OU                                                                                    : amp*meter**-2 
    """ 
    eqs_IO = eqs_IO_beta
    eqs_IO += eqs_IO_alpha
    eqs_IO += eqs_IO_tau
    eqs_IO += eqs_IO_inf
    eqs_IO += eqs_IO_activation
    eqs_IO += eqs_IO_Iax
    eqs_IO += eqs_IO_Iden
    eqs_IO += eqs_IO_Isom
    eqs_IO += eqs_IO_Ca
    if f0 == 'eye_blink' or f0 == 'pulse' or f0 == 'eye_blink_open' or f0 == 'pulse_open':
        eqs_IO += eqs_IO_V_copy
        print(f'{f0} IO_OU_Copy added')
    else: eqs_IO += eqs_IO_V
    eqs_IO += eqs_vector
    eqs_IO += eqs_noise
    return eqs_IO                
                



