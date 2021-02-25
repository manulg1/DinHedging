# -*- coding: utf-8 -*-
"""
Dinamic Hedging - Multiprocessing

@author: Manuel Luna
"""

#Required packages
import numpy as np
import pandas as pd
import time
import multiprocessing


from sim_cor_p import random_simulation, cor_simulation, simulation_results
from DinInsurance import NPV_production, option, growth_model


# The process have been divided in three parts : Data Collection, Main Function and Testing process

'''
DATA COLLECTION
'''


# time horizon - months
th = 12
risk_free = 0.015*(1/12)
simulations = 200

'''
Price Index
'''
# Read Salmon Index data
index = pd.read_csv('Data/Salmon_monthlyprices.csv', encoding = "ISO-8859-1")

ix_prices = index.iloc[0:len(index)-th,1:10]
test_prices = index.iloc[len(index)-th-1:len(index),0:10]

# Ln Returns df 
ix_returns = np.log(ix_prices) - np.log(ix_prices.shift(1))
ix_returns = ix_returns.replace([np.inf, -np.inf], np.nan)

# Correlation
corr = ix_prices.tail(48).corr()


'''
Water Temperature Data
'''

# Read the sea temperatures Dataset
temp = pd.read_csv('Data/NorTemp.csv', encoding = "ISO-8859-1")

# Select the location and time horizon -> Avg Data 2013 - 2018. Nord-Trøndelag
avg_temp = temp.iloc[48:len(temp)-th+6,[1,5]].reset_index().iloc[:,1:3]
test_temp = temp.iloc[len(temp)-th:len(temp),[1,5]].reset_index().iloc[:,1:3]

avg_temp.columns = ['Month', 'Temperature']
test_temp.columns = ['Month', 'Temperature']

# Calculate the Running Avg
avg_temp['Run_avg'] = avg_temp['Temperature']*np.nan
for t in range(6,len(avg_temp['Temperature'])-6):
    a = np.mean([avg_temp['Temperature'][t-6],avg_temp['Temperature'][t-5],avg_temp['Temperature'][t-4],avg_temp['Temperature'][t-3], avg_temp['Temperature'][t-2], avg_temp['Temperature'][t-1], avg_temp['Temperature'][t], avg_temp['Temperature'][t+1],avg_temp['Temperature'][t+2],avg_temp['Temperature'][t+3],avg_temp['Temperature'][t+4],avg_temp['Temperature'][t+5]])
    b = np.mean([avg_temp['Temperature'][t-5],avg_temp['Temperature'][t-4],avg_temp['Temperature'][t-3],avg_temp['Temperature'][t-2], avg_temp['Temperature'][t-1], avg_temp['Temperature'][t], avg_temp['Temperature'][t+1],avg_temp['Temperature'][t+2],avg_temp['Temperature'][t+3],avg_temp['Temperature'][t+4],avg_temp['Temperature'][t+5],avg_temp['Temperature'][t+6]])
    avg_temp['Run_avg'][t] = np.mean([a,b])


avg_temp=avg_temp.iloc[6:len(avg_temp)-6,:]

# and the set of addicional degrees by month
add_m = avg_temp.groupby(avg_temp.Month).mean()
add_m['Temperature'] = add_m['Temperature']/np.mean(add_m['Temperature'])
mult1= [add_m[add_m.index==test_temp.Month[0]].Temperature.item(),add_m[add_m.index==test_temp.Month[1]].Temperature.item(),add_m[add_m.index==test_temp.Month[2]].Temperature.item(),add_m[add_m.index==test_temp.Month[3]].Temperature.item(),add_m[add_m.index==test_temp.Month[4]].Temperature.item(),add_m[add_m.index==test_temp.Month[5]].Temperature.item(),add_m[add_m.index==test_temp.Month[6]].Temperature.item(),add_m[add_m.index==test_temp.Month[7]].Temperature.item(),add_m[add_m.index==test_temp.Month[8]].Temperature.item(),add_m[add_m.index==test_temp.Month[9]].Temperature.item(),add_m[add_m.index==test_temp.Month[10]].Temperature.item(),add_m[add_m.index==test_temp.Month[11]].Temperature.item()]


# Returns temp
#temp_returns = avg_temp.pct_change()

temp_returns = np.log(avg_temp['Run_avg']) - np.log(avg_temp['Run_avg'].shift(1))
temp_returns = temp_returns.replace([np.inf, -np.inf], np.nan)


'''
SIMULATION FUNCTION
'''


def dinamic_hedging(ix_prices, ix_returns, corr, avg_temp, temp_returns, numb):
    
    # Initial data could be variables or parameters
    th = 12
    risk_free = 0.015*(1/12)
    simulations = 200
    tho = th + 1
    
    # Inicial Price Simulation
    psim_i = cor_simulation(ix_returns,ix_prices)
    psim_i.monte_carlo_cor_sim(simulations, th)
    

    # Inicial Temp Simulation
    tsim_i = random_simulation(temp_returns, avg_temp['Run_avg']
    tsim_i.monte_carlo(simulations, tho)
    
    mult = []
    for j in range(0,tho):
        mult2 = (tho*100)*mult1
        mult.append(mult2[j])
    
    tsim_i.simulation_df = tsim_i.simulation_df.iloc[1:tho+1,:].mul(mult, axis=0)
    tsim_i.simulation_df=tsim_i.simulation_df.reset_index().iloc[:,1:(simulations+1)]
    
    
    # Growth model for 1000 juveniles of 100gr
    gm_i = growth_model( 0.150, 10000)
    gm_i.aq_simulation(tsim_i.simulation_df, 0)
    gm_i.mk_simulation(price_sim = psim_i.results_mc)
    
                              ## The proces is first carried ou for the initial situation, then repeated for the following steps and finally adapted for the last observation.
                               
    # Prepare Dataframe to note the results. 
    real_df = gm_i.mk_result.copy()
    real_df = pd.concat([real_df, gm_i.sel_prices.copy()], axis=1)
    
    # STARTING CONDITIONS
    # Production Value
    res1 = NPV_production(real_df.iloc[0,2], real_df.iloc[0,3], tsim_i.simulation_df, psim_i.results_mc, 0)
    real_df.loc[0,'Final weight'] = res1.total_weight
    real_df.loc[0,'Mean Value'] = res1.mean_npv
    real_df.loc[0,'Vol'] = res1.std_npv/res1.mean_npv
    # Option
    res2 = option(res1.mean_npv, 180000, th/12, 0.01, res1.std_npv/res1.mean_npv, 0)
    res2.put_price_delta()
    real_df.loc[0,'Hedge Cost'] = res2.p_price
    real_df.loc[0,'Hedge Ratio'] = res2.p_delta
    # Dinamic hedge
    real_df.loc[0,'Secure Value'] = real_df.loc[0,'Mean Value']*(-real_df.loc[0,'Hedge Ratio'])
    real_df.loc[0,'Risk Position'] = real_df.loc[0,'Mean Value']*(1+real_df.loc[0,'Hedge Ratio'])
    
    real_df.loc[0,'Futures Price'] = real_df.loc[0,'Price']*(1+risk_free)
    real_df.loc[0,'Futures Quantity'] = real_df.loc[0,'Secure Value']/real_df.loc[0,'Futures Price']
    
    a=real_df.loc[0,'Hedge Cost']
    real_df.loc[0,'Economic Result'] = 0
    real_df.loc[0,'Cash Flow Position'] = real_df.loc[0,'Economic Result']+(real_df.loc[0,'Secure Value']/(1+risk_free))+real_df.loc[0,'Hedge Cost']
    
    
    
    # FOLLOWING STEPS
    new_temp = avg_temp['Run_avg'].reset_index().iloc[:,1].copy()
    sel_prices = gm_i.sel_prices.copy()
    new_prices = ix_prices.copy()
    
    for i in range(1,len(real_df)-1):
       
        new_temp[len(new_temp)]=real_df.iloc[i-1,1]*(1/mult[i-1]) # El dato de temp del paso anterior sin la est, se usa en este como último dato real para después simular
        new_temp_returns = np.log(new_temp) - np.log(new_temp.shift(1))
        tsim_b = random_simulation(new_temp_returns, new_temp)
        tsim_b.monte_carlo(simulations, tho-i)
        multi=mult[i:]
        tsim_b.simulation_df = tsim_b.simulation_df.iloc[1:].mul(multi, axis=0)
        tsim_b.simulation_df=tsim_b.simulation_df.reset_index().iloc[:,1:(simulations+1)]

        new_prices = new_prices.append(sel_prices.iloc[i-1,:]).reset_index().iloc[:,1:10]
        new_prices_returns = np.log(new_prices) - np.log(new_prices.shift(1))
    
        psim_b = cor_simulation(new_prices_returns,new_prices)
        psim_b.monte_carlo_cor_sim(simulations, th-i)
        
        # Production Value
        res1 = NPV_production(real_df.iloc[i,2], real_df.iloc[i,3], tsim_b.simulation_df, psim_b.results_mc, i)
        real_df.loc[i,'Final weight'] = res1.total_weight
        real_df.loc[i,'Mean Value'] = res1.mean_npv
        real_df.loc[i,'Vol'] = res1.std_npv/res1.mean_npv
        # Option
        res2 = option(res1.mean_npv, 180000, (th-i)/12, 0.01, res1.std_npv/res1.mean_npv, 0)
        res2.put_price_delta()
        real_df.loc[i,'Hedge Cost'] = real_df.loc[i-1,'Hedge Cost']*(1+risk_free)
        real_df.loc[i,'Hedge Ratio'] = res2.p_delta
        # Dinamic hedge
        real_df.loc[i,'Secure Value'] = real_df.loc[i,'Mean Value']*(-real_df.loc[i,'Hedge Ratio'])
        real_df.loc[i,'Risk Position'] = real_df.loc[i,'Mean Value']*(1+real_df.loc[i,'Hedge Ratio'])
        
        real_df.loc[i,'Futures Price'] = real_df.loc[i,'Price']*(1+risk_free)
        real_df.loc[i,'Futures Quantity'] = real_df.loc[i,'Secure Value']/real_df.loc[i,'Futures Price']
        
        if real_df.loc[i-1,'Fish Weight'] < 2:
            price_cf = real_df.loc[i,'1-2']
        elif real_df.loc[i-1,'Fish Weight'] < 3:
           price_cf = real_df.loc[i,'2-3']
        elif real_df.loc[i-1,'Fish Weight'] < 4:
           price_cf = real_df.loc[i,'3-4']
        elif real_df.loc[i-1,'Fish Weight'] < 5:
           price_cf = real_df.loc[i,'4-5']
        elif real_df.loc[i-1,'Fish Weight'] < 6:
           price_cf = real_df.loc[i,'5-6']
        elif real_df.loc[i-1,'Fish Weight'] < 7:
           price_cf = real_df.loc[i,'6-7']
        elif real_df.loc[i-1,'Fish Weight'] < 8:
           price_cf = real_df.loc[i,'7-8']
        elif real_df.loc[i-1,'Fish Weight'] < 9:
           price_cf = real_df.loc[i,'8-9']
        else:
           price_cf = real_df.loc[i,'9+']
        
        a=(a+real_df.loc[i-1,'Economic Result'])*(1+risk_free)
        real_df.loc[i,'Economic Result'] = real_df.loc[i-1,'Economic Result']*(1+risk_free)+(real_df.loc[i-1,'Futures Price']-price_cf)*real_df.loc[i-1,'Futures Quantity']
        real_df.loc[i,'Cash Flow Position'] = real_df.loc[i,'Economic Result']+(real_df.loc[i,'Secure Value']/(1+risk_free))+a
    
    
    # LAST ROWS
    real_df.loc[th,'Secure Value'] = 0
    real_df.loc[th,'Risk Position'] = real_df.loc[i,'Mean Value']
    
    if real_df.loc[i-1,'Fish Weight'] < 2:
            price_cf = real_df.loc[i,'1-2']
    elif real_df.loc[i-1,'Fish Weight'] < 3:
       price_cf = real_df.loc[i,'2-3']
    elif real_df.loc[i-1,'Fish Weight'] < 4:
       price_cf = real_df.loc[i,'3-4']
    elif real_df.loc[i-1,'Fish Weight'] < 5:
       price_cf = real_df.loc[i,'4-5']
    elif real_df.loc[i-1,'Fish Weight'] < 6:
       price_cf = real_df.loc[i,'5-6']
    elif real_df.loc[i-1,'Fish Weight'] < 7:
       price_cf = real_df.loc[i,'6-7']
    elif real_df.loc[i-1,'Fish Weight'] < 8:
       price_cf = real_df.loc[i,'7-8']
    elif real_df.loc[i-1,'Fish Weight'] < 9:
       price_cf = real_df.loc[i,'8-9']
    else:
       price_cf = real_df.loc[i,'9+']
    
    real_df.loc[th,'Economic Result'] = real_df.loc[th-1,'Economic Result']*(1+risk_free)+(real_df.loc[th-1,'Futures Price']-price_cf)*real_df.loc[th-1,'Futures Quantity']+(real_df.loc[th,'Total Weight'])*real_df.loc[th,'Price']
    real_df.loc[th,'Cash Flow Position'] = real_df.loc[th,'Economic Result']+real_df.loc[0,'Hedge Cost']*((1+risk_free)**th)
    
    nombre = 'Results/results_'+str(numb)+'.npy'
    np.save(nombre,real_df)
    
    return real_df


'''
TESTING PROCESS
'''

# The code is prepared for multiprocessing (running independent parallel processes) due to its computational requirements

# Test in 150 different scenarios
df_datos=(range(0,150))

numbers = range(len(df_datos))

result = []


def mp_worker(number):
    dato = df_datos[number]
    res = dinamic_hedging(ix_prices, ix_returns, corr, avg_temp, temp_returns, dato)
    return res

def mp_handler():
    p = multiprocessing.Pool(3)
    result = p.map(mp_worker, numbers)
    return result

if __name__ == '__main__':
    start_time = time.time()
    result_fin = mp_handler()
    end_time = time.time() - start_time
    print(f"Processing {len(df_datos)} numbers took {end_time} time using multiprocessing.")
    print(result_fin)
