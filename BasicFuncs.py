
'''
Provide basic functions for reading the data

Maximilian Eggl
Nov 2024
'''


import numpy as np
import os
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def PrepData(cdata):

    Cuts =  np.cumsum([x.shape[1] for x in cdata["Data"]["C_unsorted"]])

    sdat = np.split(cdata["Data"]["S_all"],Cuts,axis=1)

    sdat.pop(-1)
    cdat = np.split(cdata["Data"]["C_all"],Cuts,axis=1)

    cdat.pop(-1)
    craw = np.split(cdata["Data"]["C_Raw_all"],Cuts,axis=1)

    craw.pop(-1)

    return sdat,cdat,craw

def ReadExpt_N(ExptNum,Dir):
    csvDir = [f for f in os.listdir(Dir) if f.startswith(str(ExptNum)+'.')][0]
    try:
        Fn = Dir+csvDir+'/'+[f for f in os.listdir(Dir + csvDir) if f.endswith('.csv')][0]
        T = []
        Chan = []
        Val =  []
        with open(Fn, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                T.append(row[0])
                Chan.append(row[1])
                Val.append(row[2])

        Chan = np.array(Chan)
        T    = np.array(T)
        Val  = np.array(Val)

        T1 = T[np.where(Chan == ' GPIO-1')]
        V1 = Val[np.where(Chan == ' GPIO-1')]

        T2 = T[np.where(Chan == ' GPIO-2')]
        V2 = Val[np.where(Chan == ' GPIO-2')]

        T3 = T[np.where(Chan == ' GPIO-3')]
        V3 = Val[np.where(Chan == ' GPIO-3')]

        T4 = T[np.where(Chan == ' GPIO-4')]
        V4 = Val[np.where(Chan == ' GPIO-4')]

        T1 = T1.astype(float)
        V1[V1==' nan'] = 0
        V1 = V1.astype(int)
        T2 = T2.astype(float)
        
        V2[V2==' nan'] = 0
        V2 = V2.astype(int)
        T3 = T3.astype(float)
        
        V3[V3==' nan'] = 0
        V3 = V3.astype(int)
        T4 = T4.astype(float)
        
        V4[V4==' nan'] = 0
        V4 = V4.astype(int)
    except:
        pass
    
    return [[T1,V1],[T2,V2],[T3,V3],[T4,V4]]

def SpikeLoc(x,thresh=100):
    
    xdiff = x[1:] - x[0:-1]
    xdiff_mean = np.abs(xdiff).mean()
    spikes = xdiff > abs(xdiff_mean)+thresh
    
    y = np.linspace(0,len(x)-2,len(x)-1)
    
    for t in y[spikes]:
        if(spikes[int(t)-10:int(t)].any() or spikes[int(t)+1:int(t)+11].any()):
            spikes[int(t)] =  False
    return spikes

def SoundTimes(V):
    SoundTimes_on = []
    for v in V:
        SoundTimes_on.append(v[0][:-1][SpikeLoc(v[1])])

    SoundTimes_off = []
    for v in V:
        SoundTimes_off.append(v[0][:-1][SpikeLoc(-v[1])])
    
    return SoundTimes_on,SoundTimes_off

#======================================================================
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    avg = ret[n - 1:] / n
    var = np.cumsum(a[n-1:]-avg, dtype=float)/(n-1)
    return avg,var
#======================================================================
def SpikeCount(x,thresh=50):
    localMax = argrelextrema(x, np.greater)[0]
    spike = []
    while len(localMax)>0:
        t = localMax[0]
        if(np.logical_and(abs(localMax-t)<thresh,abs(localMax-t)>0).any()):
            m = np.where(abs(localMax-t)<thresh)
            spike.append(localMax[m][np.argmax(x[localMax[m]])])
            localMax = np.delete(localMax,m)
        else:
            spike.append(t)
            localMax = np.delete(localMax,0)
    return spike

def SpikeLoc_zscore2(x,z_sample,thresh=3):
    m = np.mean(z_sample)
    sig = np.std(z_sample)
    z_score = (x-m)/sig
    localMax = argrelextrema(x, np.greater)[0]
    spike = []
    while len(localMax)>0:
        t = localMax[0]
        if(np.logical_and(abs(localMax-t)<50,abs(localMax-t)>0).any()):
            m = np.where(abs(localMax-t)<50)
            spike.append(localMax[m][np.argmax(x[localMax[m]])])
            localMax = np.delete(localMax,m)
        else:
            spike.append(t)
            localMax = np.delete(localMax,0)
        if(z_score[spike[-1]]<thresh):
            spike.pop(-1)
                    
    return spike

def z_score(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(X.T).T
    return Xz

def GenDicts(Mat,datDir):
    VDict  =  dict((n,i+2) for i,n in zip(np.arange(0,10),Names[1:-1]))
    CDict  =  dict((n,i) for i,n in zip(np.arange(0,11),Names))
    SDict  =  dict((n,i) for i,n in zip(np.arange(0,11),Names))
    S2Dict =  dict((n,i) for i,n in zip(np.arange(0,11),Names))
    TDict  =  dict((n,i) for i,n in zip(np.arange(0,11),Names))
    PDict  =  dict((n,i+2) for i,n in zip(np.arange(0,10),Names[1:-1]))

    data = read_mat(Mat)
    sdat,cdat = PrepData(data)

    for i,(k,c,s) in enumerate(tqdm(zip(Names,cdat,sdat))):

        if(i==0 or i==10):
            TDict[k] = np.linspace(0,60,c.shape[1])
        else:
            V = ReadExpt_N(VDict[k],datDir)
            VDict[k] = V
            TDict[k] = np.linspace(VDict[k][0][0][0],VDict[k][0][0][-1],c.shape[1])
        CDict[k] = c
        l = []
        for c1 in c:
            l.append(SpikeCount(c1))
        S2Dict[k] = l
        s0 = np.argwhere(s>0)
        l = [[] for _ in range(s0[:,0].max()+1)]
        for s1 in s0:
            if(set(np.arange(s1[1]-10,s1[1]+10)).isdisjoint(l[s1[0]])):
                l[s1[0]].append(TDict[k][s1[1]])
        SDict[k] = l

        if(i>0 and i<10):
            S_on,S_off = SoundTimes(V)
            d = {}
            for j,p in enumerate(Phases):
                for i,(s_on,s_off) in enumerate(zip(S_on[j],S_off[j])):
                   d[p+'_'+str(i)] = (s_on,s_off) 
            PDict[k] = d
    return VDict,CDict,SDict,S2Dict,TDict,PDict

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=20, maxasterix=1,offS=0.6,ax=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if(np.round(data,3)==0):
            text = r' $p<0.0005$'        
        else:
            text = r' $p$: ' + str(np.round(data,3))
    
    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    if(ax==None):
        ax_y0, ax_y1 = plt.gca().get_ylim()
    else:
        ax_y0, ax_y1 = ax.get_ylim()

    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)


    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs
    if(ax==None):
        plt.plot(barx, bary, c='black')
        plt.text(*mid, text, **kwargs)
    else:
        ax.plot(barx, bary, c='black')
        ax.text(*mid, text, **kwargs)
    return max(ly, ry)

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def FindCorr(Expt1,Expt2,C,thresh,plot=False):
    Corr = np.corrcoef(C[Expt1])
    Strong_Corr = np.argwhere(np.logical_and(Corr<0.995,abs(Corr)>thresh))
   # print('These are strongly correlated:',Strong_Corr)
    if(plot):
        for (i,j) in np.unique(np.sort(Strong_Corr,axis=1),axis=0):
            plt.plot(C[Expt1][i])
            plt.plot(C[Expt1][j],'r')
            plt.show()
    
    OtherList = np.argwhere(np.isnan(C[Expt2].sum(axis=1)))   
    
    exist = 0
    for s in np.unique(np.sort(Strong_Corr,axis=1),axis=0):
        if(not(s[0] in OtherList or s[1] in OtherList)):
            tCf = np.corrcoef([C[Expt2][s[0]],C[Expt2][s[1]]])[1,0]
            if(abs(tCf)>thresh):
                print(f'These are common {Expt1}: {s} with correlation: {Corr[s[0],s[1]]:.3f}')
                print(f'In {Expt2} they are correlated:{tCf:.3f}')
                fig,ax = plt.subplots(1,2,figsize=(10,5))
                ax[0].plot(C[Expt1][s[0]])
                ax[0].plot(C[Expt1][s[1]])
                ax[1].plot(C[Expt2][s[0]])
                ax[1].plot(C[Expt2][s[1]])
                plt.show()
            else:
                pass
            
            exist = 1

def ReadCalc(Dir, Expts, Type):
    if(os.path.exists(Dir+'/C_'+Type+'.npy')):
        print('Calcium data already exists')
        C = np.load(Dir+'/C_'+Type+'.npy', allow_pickle=True)
        V = np.load(Dir+'/V_'+Type+'.npy', allow_pickle=True)
        S = np.load(Dir+'/S_'+Type+'.npy', allow_pickle=True)
        S2 = np.load(Dir+'/S2_'+Type+'.npy', allow_pickle=True)
        P = np.load(Dir+'/P_'+Type+'.npy', allow_pickle=True)
        T = np.load(Dir+'/T_'+Type+'.npy', allow_pickle=True)
    else:
        V = []
        C = []
        S = []
        S2 = []
        T = []
        P = []
        if(Type == 'NA'):
            Name = 'Nucleus Accumbens/'
        else:
            Name = 'Medium Prefrontal Cortex/'
        for e in tqdm(Expts):
            datDir = Dir+'/'+e+'/'+e+'_' + Name
            datMat = 'Data_Miniscope_PP_new_'+e+'.mat'
            v, c, s, s2, t, p = GenDicts(datDir+datMat, datDir)
            V.append(v),
            C.append(c)
            S.append(s)
            S2.append(s2)
            T.append(t)
            P.append(p)

        np.save(Dir+'/C_'+Type, C)
        np.save(Dir+'/V_'+Type, V)
        np.save(Dir+'/S_'+Type, S)
        np.save(Dir+'/S2_'+Type, S2)
        np.save(Dir+'/P_'+Type, P)
        np.save(Dir+'/T_'+Type, T)
    return C, V, S, S2, P, T


def ReadBehav(Dir, Expts, Type):
    NameConv = {'Extinction 1': 'ED1', 'Extinction 2': 'ED2', 'Extinction 3': 'ED3', 'Extinction 4': 'ED4',
                'Extinction Retrieval 1': 'ER1', 'Extinction Retrieval 2': 'ER2', 'C2': 'C2', 'C3': 'C3',
                'Renewal': 'Re'}
    if(os.path.exists(Dir+'/bDat.npy')):
        print('Behavioural data already exists')
        bData = np.load(Dir+'/bDat.npy', allow_pickle=True)
    else:
        bData = []
        for expt in Expts:
            L = os.listdir(Dir+'/'+expt+'/')
            T = {}
            for l in tqdm(sorted(L)):
                if('.xlsx' in l):
                    print(l)
                    if(Type == 'NA'):
                        p = pd.read_excel(Dir+'/'+expt+'/'+l, skiprows=38)
                        if 'cm' in p.keys():
                            p = pd.read_excel(Dir+'/'+expt+'/'+l, skiprows=39)
                    else:
                        p = pd.read_excel(Dir+'/'+expt+'/'+l, skiprows=39)
                        if 'cm' in p.keys():
                            p = pd.read_excel(Dir+'/'+expt+'/'+l, skiprows=38)

                    Mobility = np.array(p['Mobility state(Mobile)'][1:])
                    iMobility = np.array(p['Mobility state(Immobile)'][1:])
                    Time = np.array(p['Trial time'][1:])
                    x_center = np.array(p['X center'][1:])
                    y_center = np.array(p['Y center'][1:])
                    try:
                        ASA = np.array(p['Activity state(Highly active)'][1:])
                        ASI = np.array(p['Activity state(Inactive)'][1:])
                    except:
                        ASA = np.array(p['Activity State(Highly active)'][1:])
                        ASI = np.array(p['Activity State(Inactive)'][1:])
                    try:
                        ASA2 = np.array(
                            p['Activity state 2(Highly active)'][1:])
                        ASI2 = np.array(p['Activity state 2(Inactive)'][1:])
                    except:
                        ASA2 = np.array(
                            p['Activity State 2(Highly active)'][1:])
                        ASI2 = np.array(p['Activity State 2(Inactive)'][1:])

                    hMobility = np.array(
                        p['Mobility state(Highly mobile)'][1:])

                    t = np.vstack([Time, x_center, y_center, Mobility, iMobility,
                                  ASA, ASI, ASA2, ASI2, hMobility])

                    t = np.delete(t, np.any(t == '-', axis=0),
                                  axis=1).astype(np.double)

                    for k in NameConv.keys():
                        if(k in l):
                            break
                    print(k)
                    T[NameConv[k]] = t
            bData.append(T)
        np.save(Dir+'/bDat.npy', bData)
    return bData