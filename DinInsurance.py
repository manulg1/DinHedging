# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:11:19 2019

@author: Manuel Luna
"""

import numpy as np
import math
import pandas as pd
from scipy import stats
import random


'''
Production Value
'''
class NPV_production:
    def __init__(self, inicial_weight, inicial_fishes, temp_sim, price_sim, day):
        self.in_w = inicial_weight
        self.in_f = inicial_fishes
        self.temp_sim = temp_sim
        self.price_sim = price_sim
        self.day = day
        
        gm_t = growth_model(self.in_w, self.in_f)
        tw = []
        npv = []
        for i in range(0,1000):
            gm_t.aq_simulation(self.temp_sim, self.day)
            gm_t.mk_simulation(self.price_sim)
            tw.append(gm_t.mk_result['Total Weight'].iloc[-1])
            npv.append(gm_t.mk_result['Total Value'].iloc[-1]/(1+0.015*len(gm_t.mk_result['Total Value'])/12)) #VAN
        self.total_weight = np.mean(tw)
        self.mean_npv = np.mean(npv)
        self.std_npv = np.std(npv)
        

class option:
    """
    This class will group the different black-shcoles calculations for an opion
        
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset

    """
    def __init__(self, s, k, t, rf, vol, div):
        self.k = k
        self.s = s
        self.rf = rf
        self.vol = vol
        self.t = t
        if self.t == 0: self.t = 0.000001 ## Case valuation in expiration date
        self.div = div

 
    def call_price_delta(self):
        d1 = ( math.log(self.s/self.k) + ( self.rf + self.div + math.pow( self.vol, 2)/2 ) * self.t ) / ( self.vol * math.sqrt(self.t) )
        d2 = d1 - self.vol * math.sqrt(self.t)
        calc_price = ( stats.norm.cdf(d1) * self.s * math.exp(-self.div*self.t) - stats.norm.cdf(d2) * self.k * math.exp( -self.rf * self.t ) )
        delta = stats.norm.cdf(d1)
        
        self.c_price = calc_price
        self.c_delta = delta
   
    def put_price_delta(self):
        d1 = ( math.log(self.s/self.k) + ( self.rf + self.div + math.pow( self.vol, 2)/2 ) * self.t ) / ( self.vol * math.sqrt(self.t) )
        d2 = d1 - self.vol * math.sqrt(self.t)
        calc_price =  ( -stats.norm.cdf(-d1) * self.s * math.exp(-self.div*self.t) + stats.norm.cdf(-d2) * self.k * math.exp( -self.rf * self.t ) )
        delta = -stats.norm.cdf(-d1)
        
        self.p_price = calc_price
        self.p_delta = delta




'''
Growth
'''
class growth_model:
    def __init__(self, inicial_weight, inicial_fishes):
        self.in_weight = inicial_weight
        self.in_fish = inicial_fishes

    # basic only uses the asset’s daily volatility
    def aq_simulation(self, temperatures, day):
        temp = temperatures
        in_weight = self.in_weight
        in_fish = self.in_fish
        self.day = day
        
        s=random.randint(0,len(temp.columns)-1)
        
        result = pd.DataFrame()
        result['Month'] = range(0,len(temp))
        result['Temperature'] = temp.iloc[:,s]
        
        result['Fish Weight'] = in_weight
        result['Fish Number'] = in_fish
        for i in range(1,len(temp)):
            if result.loc[i-1,'Fish Weight'] <= 7:
                result.loc[i,'Fish Weight'] = result.loc[i-1,'Fish Weight']*(1+((0.037*result.loc[i-1,'Temperature'])/(1+math.exp(0.037*result.loc[i-1,'Temperature']*(result.loc[i-1,'Month']-17.037)))))
            else:
                result.loc[i,'Fish Weight'] = result.loc[i-1,'Fish Weight']
            result.loc[i,'Fish Number'] = result.loc[i-1,'Fish Number']*(1-(0.005+0.0015*result.loc[i-1,'Month']))
        result['Total Weight'] = result['Fish Weight']*result['Fish Number'] 
        
        self.aq_result = result

    # basic only uses the asset’s daily volatility
    def mk_simulation(self, price_sim=False, price_actual=False):
        result = self.aq_result
        
        if price_sim==False:
            sel_prices=price_actual
            # The price is settled for each month depending on the class at this moment
            result['Price'] = price_actual.iloc[0,0]
            for i in range(0,len(result)):
                if result.loc[i,'Fish Weight'] < 2:
                   result.loc[i,'Price'] = price_actual.iloc[i,0]
                elif result.loc[i,'Fish Weight'] < 3:
                   result.loc[i,'Price'] = price_actual.iloc[i,1]
                elif result.loc[i,'Fish Weight'] < 4:
                   result.loc[i,'Price'] = price_actual.iloc[i,2]
                elif result.loc[i,'Fish Weight'] < 5:
                   result.loc[i,'Price'] = price_actual.iloc[i,3]
                elif result.loc[i,'Fish Weight'] < 6:
                   result.loc[i,'Price'] = price_actual.iloc[i,4]
                elif result.loc[i,'Fish Weight'] < 7:
                   result.loc[i,'Price'] = price_actual.iloc[i,5]
                elif result.loc[i,'Fish Weight'] < 8:
                   result.loc[i,'Price'] = price_actual.iloc[i,6]
                elif result.loc[i,'Fish Weight'] < 9:
                   result.loc[i,'Price'] = price_actual.iloc[i,7]
                else:
                   result.loc[i,'Price'] = price_actual.iloc[i,8]
        else:
            # The simulation used is chosen randomly
            s=random.randint(0,len(price_sim[0].columns)-1)
            self.ch_sim=s
            # The price is settled for each month depending on the class at this moment
            result['Price'] = price_sim[0].iloc[0,s]
            for i in range(0,len(result)):
                if result.loc[i,'Fish Weight'] < 2:
                   result.loc[i,'Price'] = price_sim[0].iloc[i,s]
                elif result.loc[i,'Fish Weight'] < 3:
                   result.loc[i,'Price'] = price_sim[1].iloc[i,s]
                elif result.loc[i,'Fish Weight'] < 4:
                   result.loc[i,'Price'] = price_sim[2].iloc[i,s]
                elif result.loc[i,'Fish Weight'] < 5:
                   result.loc[i,'Price'] = price_sim[3].iloc[i,s]
                elif result.loc[i,'Fish Weight'] < 6:
                   result.loc[i,'Price'] = price_sim[4].iloc[i,s]
                elif result.loc[i,'Fish Weight'] < 7:
                   result.loc[i,'Price'] = price_sim[5].iloc[i,s]
                elif result.loc[i,'Fish Weight'] < 8:
                   result.loc[i,'Price'] = price_sim[6].iloc[i,s]
                elif result.loc[i,'Fish Weight'] < 9:
                   result.loc[i,'Price'] = price_sim[7].iloc[i,s]
                else:
                   result.loc[i,'Price'] = price_sim[8].iloc[i,s]
            # Fill dataset with all prices and clases selected
            sel_prices = pd.DataFrame()
            sel_prices['1-2'] = price_sim[0].iloc[:,s]
            sel_prices['2-3'] = price_sim[1].iloc[:,s]
            sel_prices['3-4'] = price_sim[2].iloc[:,s]
            sel_prices['4-5'] = price_sim[3].iloc[:,s]
            sel_prices['5-6'] = price_sim[4].iloc[:,s]
            sel_prices['6-7'] = price_sim[5].iloc[:,s]
            sel_prices['7-8'] = price_sim[6].iloc[:,s]
            sel_prices['8-9'] = price_sim[7].iloc[:,s]
            sel_prices['9+'] = price_sim[8].iloc[:,s]
        result['Total Value'] = result['Total Weight']*result['Price']
        self.mk_result = result
        self.sel_prices = sel_prices

