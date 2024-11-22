
'''
Provide extended for reading the data

Maximilian Eggl
Nov 2024
'''

from tqdm.notebook import tqdm, trange
import os

import numpy as np
import pandas as pd
import scipy as sp
from pymatreader import read_mat

import math
from BasicFuncs import *

def GetFreezing(datDir):
    b1 = False
    b2 = False
    for file in os.listdir(datDir):
        if file.endswith(".xlsx"):
            BDat = pd.read_excel(datDir+'/'+file,
            sheet_name= 'Track-Arena 1-Subject 1', skiprows=36)
            if("Unnamed: 1" in BDat.keys()):
                BDat = pd.read_excel(datDir+'/'+file,
                sheet_name= 'Track-Arena 1-Subject 1', skiprows=37)
            BDat = BDat.drop(0)
            T = np.array(BDat.replace('-',np.nan).dropna()['Recording time'])
            BDat = np.array(BDat.replace('-',np.nan).dropna()['Immobility'])
            b1 = True
        if file.endswith(".csv"):
            GDat = pd.read_csv(datDir+'/'+file)
            G1 = GDat[GDat[' Channel Name'] == ' GPIO-1']
            G4 = GDat[GDat[' Channel Name'] == ' GPIO-4']
            
            try:
                ShockTime = [np.where(G4[' Value']>100)[0][0]]
                PostShock = np.where(G4[' Value'][ShockTime[0]:]<100)[0][0]+ShockTime[0]
                ShockTime.append(np.where(G4[' Value'][PostShock:]>100)[0][0]+PostShock)
                ShockTime = [G4['Time (s)'].iloc[s] for s in ShockTime]
            except Exception as e:
                ShockTime = [math.nan,math.nan]
                
            try:
                StartTime,EndTime = G1['Time (s)'][np.where(G1[' Value']>100)[0][[0,-1]]]
                Tmax = G1['Time (s)'].max()
            except Exception as e:
                EndTime,StartTime,Tmax = math.nan,math.nan,math.nan
            b2 = True
        if(b1 and b2):
            return [np.array(T).astype(np.float64),BDat],(StartTime,EndTime,Tmax),ShockTime

def GetFreezingNew(datDir):
    b1 = False
    b2 = False
    for file in os.listdir(datDir):
        if file.endswith(".xlsx"):
            BDat = pd.read_excel(datDir+'/'+file,
            sheet_name= 'Track-Arena 1-Subject 1', skiprows=38)
            BDat = BDat.drop(0)
            T = np.array(BDat.replace('-',np.nan).dropna()['Recording time'])
            BDat = np.array(BDat.replace('-',np.nan).dropna()['Mobility state(Immobile)'])
            b1 = True
        if file.endswith(".csv"):
            GDat = pd.read_csv(datDir+'/'+file)
            G1 = GDat[GDat[' Channel Name'] == ' GPIO-1']
            G4 = GDat[GDat[' Channel Name'] == ' GPIO-4']
            
            try:
                ShockTime = [np.where(G4[' Value']>100)[0][0]]
                PostShock = np.where(G4[' Value'][ShockTime[0]:]<100)[0][0]+ShockTime[0]
                ShockTime.append(np.where(G4[' Value'][PostShock:]>100)[0][0]+PostShock)
                ShockTime = [G4['Time (s)'].iloc[s] for s in ShockTime]
            except Exception as e:
                ShockTime = [math.nan,math.nan]
                
            try:
                StartTime,EndTime = G1['Time (s)'][np.where(G1[' Value']>100)[0][[0,-1]]]
                Tmax = G1['Time (s)'].max()
            except Exception as e:
                EndTime,StartTime,Tmax = math.nan,math.nan,math.nan
            b2 = True
        if(b1 and b2):
            return [np.array(T).astype(np.float64),BDat],(StartTime,EndTime,Tmax),ShockTime

def FilterFreezing(f,FreezeTime):
    f2 = np.zeros_like(f)
    x = sp.signal.find_peaks(f,distance=20,plateau_size=FreezeTime/0.04)
    for x1,x2 in zip(x[1]['left_edges'],x[1]['right_edges']):
        f2[x1-1:x2+1] = 1
    return f2    

def GetDat(Batch,Expts,datDir,FreezeTime=2,eFlag = 0):

    print('='*20)
    print('Doing '+Batch)
    DatNames  = []
    Dat       = []
    Times     = []
    ShockTimes = []
    TimeCalcDat = []
    CalcDat = []
    RawDat  = []
    for file in tqdm(os.listdir(datDir+Batch)):
        if os.path.isdir(datDir+Batch+'/'+file):
            newDir = datDir+Batch+'/'+file+'/'
            DatNames.append(file)
            T = []
            d = []
            st = []
            for file2 in np.sort(os.listdir(newDir)):
                if(os.path.isdir(newDir+file2) and file2+'/' in Expts):
                    f,t,s = GetFreezing(newDir+file2+'/')
                    f2 = FilterFreezing(f[1],FreezeTime)
                    f.append(f2)
                    d.append(f)
                    T.append(t)
                    st.append(s)
            Dat.append(d)
            Times.append(T)
            ShockTimes.append(st)
            
            MatDir =datDir+Batch+'/'+file+'/Data_Miniscope_PP.mat'
            if(eFlag==0):
                NDat = read_mat(MatDir)
                sDat,cDat,cRaw = PrepData(NDat)
                CalcDat.append([cDat[1],cDat[2],cDat[3],cDat[4],cDat[-1]]) 
                RawDat.append([cRaw[1],cRaw[2],cRaw[3],cRaw[4],cRaw[-1]]) 
                lens = [len(cDat[1][0]),len(cDat[2][0]),len(cDat[3][0]),len(cDat[4][0]),len(cDat[-1][0])]
                TimeCalcDat.append([np.linspace(0,tm[-1],l) for tm,l in zip(T,lens)])
            else:
                MatDirU =datDir+Batch+'/'+file+'/Data_Miniscope_PP_upper.mat'
                flag=0
                if(os.path.isfile(MatDir)):
                    NDat = read_mat(MatDir)
                    sDat,cDat,cRaw = PrepData(NDat)
                    flag=1
                
                if(os.path.isfile(MatDirU)):
                    NDat = read_mat(MatDirU)
                    sDat,cDatU,cRawU = PrepData(NDat)
                    flag+=2
                    
                if(flag==1):
                    CalcDat.append([cDat[1],cDat[2],cDat[3],cDat[4],cDat[-1]]) 
                    RawDat.append([cRaw[1],cRaw[2],cRaw[3],cRaw[4],cRaw[-1]]) 
                elif (flag==2):
                    CalcDat.append([cDatU[0],cDatU[1],cDatU[2],cDatU[3],cDatU[4]]) 
                    RawDat.append([cRawU[1],cRawU[2],cRawU[3],cRawU[4],cRawU[-1]]) 
                elif(flag==3):
                    s0 = min(cDat[1].shape[1],cDatU[0].shape[1])
                    s1 = min(cDat[2].shape[1],cDatU[1].shape[1])
                    s2 = min(cDat[3].shape[1],cDatU[2].shape[1])
                    s3 = min(cDat[4].shape[1],cDatU[3].shape[1])
                    s4 = min(cDat[-1].shape[1],cDatU[4].shape[1])
                    CalcDat.append([np.vstack([cDat[1][:,:s0],cDatU[0][:,:s0]]),
                                    np.vstack([cDat[2][:,:s1],cDatU[1][:,:s1]]),
                                    np.vstack([cDat[3][:,:s2],cDatU[2][:,:s2]]),
                                    np.vstack([cDat[4][:,:s3],cDatU[3][:,:s3]]),
                                    np.vstack([cDat[-1][:,:s4],cDatU[4][:,:s4]])]) 
                    RawDat.append([np.vstack([cRaw[1][:,:s0],cRawU[0][:,:s0]]),
                                    np.vstack([cRaw[2][:,:s1],cDatU[1][:,:s1]]),
                                    np.vstack([cRaw[3][:,:s2],cRawU[2][:,:s2]]),
                                    np.vstack([cRaw[4][:,:s3],cRawU[3][:,:s3]]),
                                    np.vstack([cRaw[-1][:,:s4],cRawU[4][:,:s4]])]) 
                TimeCalcDat.append([np.linspace(0,tm[-1],l) for tm,l in zip(T,[s0,s1,s2,s3,s4])])
    return Dat,np.array(Times),DatNames,TimeCalcDat,CalcDat,np.array(ShockTimes),RawDat