#!/usr/bin/env python
# coding: utf-8
#author: Ryan Johnson, PHD, Alabama Water Institute
#Date: 6-6-2022


'''
Run using the OWP_env: 
https://www.geeksforgeeks.org/using-jupyter-notebook-in-virtual-environment/
https://github.com/NOAA-OWP/hydrotools/tree/main/python/nwis_client

https://noaa-owp.github.io/hydrotools/hydrotools.nwm_client.utils.html#national-water-model-file-utilities
will be benefitical for finding NWM reachs between USGS sites
'''

# Import the NWIS IV Client to load USGS site data
from hydrotools.nwis_client.iv import IVDataService
from hydrotools.nwm_client import utils
import pandas as pd
import numpy as np
import data
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_percentage_error
import hydroeval as he
import dataretrieval.nwis as nwis
##https://streamstats-python.readthedocs.io/en/latest/gallery_vignettes/plot_get_characteristics.html
import streamstats
import geopandas as gpd
from IPython.display import display
import warnings
from progressbar import ProgressBar
from datetime import timedelta
import folium
import matplotlib
import mapclassify
import time
import jenkspy
import hvplot.pandas
import holoviews as hv
from holoviews import dim, opts, streams
from bokeh.models import HoverTool
import branca.colormap as cm
import vincent
from vincent import AxisProperties, PropertySet, ValueRef, Axis
import json
from folium import features
import proplot as pplt
pplt.rc["figure.facecolor"] = "w"
import pygeohydro as gh
import pygeoutils as geoutils
from pygeohydro import NID, NWIS
import pandas as pd
from folium.plugins import StripePattern
import io
import swifter
from geopy.geocoders import Nominatim
from multiprocessing import Process

geolocator = Nominatim(user_agent="geoapiExercises")



pd.options.plotting.backend = 'holoviews'

warnings.filterwarnings("ignore")



class LULC_Eval():
    
    def __init__(self, model ,state,  startDT, endDT, cwd):
        self = self
        #self.df =df
        self.startDT = startDT
        self.endDT = endDT
        self.cwd = cwd
        self.cms_to_cfs = 35.314666212661
        self.model = model
        self.state = state
        self.cfsday_AFday = 1.983
        self.freqkeys = {
                        'D': 'Daily',
                        'M': 'Monthly',
                        'Q': 'Quarterly',
                        'A': 'Annual'
                        }
        
    def get_NWIS(self):
        print('Getting NWIS Streamstats')
        
       #Load streamstats wiht lat long to get geolocational information
        Streamstats = pd.read_hdf(self.cwd+'/Data/StreamStats/StreamStats3.h5', 'streamstats')
        Streamstats.drop_duplicates(subset = 'NWIS_site_id', inplace = True)
        Streamstats.set_index('NWIS_site_id', drop = True, inplace = True)

        self.NWIS_sites = pd.read_hdf(self.cwd+'/Data/StreamStats/StreamStats.h5', self.state)
        
        #Make all NWIS sites correct 8 digit code
        for i in np.arange(0, len(self.NWIS_sites),1):
                    self.NWIS_sites.NWIS_site_id.loc[i] = str(self.NWIS_sites.NWIS_site_id.loc[i])
                    if len(self.NWIS_sites.NWIS_site_id.loc[i]) < 8:
                        self.NWIS_sites.NWIS_site_id.loc[i] = '0' + str(self.NWIS_sites.NWIS_site_id.loc[i])
                    else:
                        self.NWIS_sites.NWIS_site_id.loc[i] = str(self.NWIS_sites.NWIS_site_id.loc[i])

        #combine streamstats to get                 
        self.NWIS_sites.set_index('NWIS_site_id', drop = True, inplace = True)
        self.NWIS_sites = pd.concat([self.NWIS_sites, Streamstats], axis=1, join="inner")

        #remove sites with not lat/long
        self.NWIS_sites = self.NWIS_sites[self.NWIS_sites['dec_lat_va'].notna()].reset_index()
        
        #drop duplicate columns
        self.NWIS_sites = self.NWIS_sites.T.reset_index()
        self.NWIS_sites = self.NWIS_sites.drop_duplicates(subset = 'index').set_index('index').T
        
        #convert to geodataframe
        self.NWIS_sites = gpd.GeoDataFrame(self.NWIS_sites, geometry=gpd.points_from_xy(self.NWIS_sites.dec_long_va, self.NWIS_sites.dec_lat_va))
        

    def get_NHD_Model_info(self):   
        print('Getting NHD reaches')
       #Get NHD reach colocated with NWIS
        self.site_id = self.NWIS_sites.NWIS_site_id
        
        NHD_reaches = []

        for site in self.site_id:
            try:
                NHD_NWIS_df = utils.crosswalk(usgs_site_codes=site)
                NHD_segment = NHD_NWIS_df.nwm_feature_id.values[0]
                NHD_reaches.append(NHD_segment)

            except:
                NHD_segment = np.nan
                NHD_reaches.append(NHD_segment)
        self.NWIS_sites['NHD_reachid'] = NHD_reaches
        
        self.NWIS_sites = self.NWIS_sites.fillna(0.0)
        
        self.NWIS_sites['NHD_reachid'] = self.NWIS_sites['NHD_reachid'].astype(int)
        
        self.NWIS_sites = self.NWIS_sites[self.NWIS_sites.NHD_reachid != 0]
        
        self.df = self.NWIS_sites.copy()
        
      


    def get_USGS_site_info(self, state):
        #url for state usgs id's
        url = 'https://waterdata.usgs.gov/'+state+'/nwis/current/?type=flow&group_key=huc_cd'

        NWIS_sites = pd.read_html(url)

        NWIS_sites = pd.DataFrame(np.array(NWIS_sites)[1]).reset_index(drop = True)

        cols = ['StationNumber', 'Station name','Date/Time','Gageheight, feet', 'Dis-charge, ft3/s']

        self.NWIS_sites = NWIS_sites[cols].dropna()
        
        self.NWIS_sites = self.NWIS_sites.rename(columns ={'Station name':'station_name', 
                                                               'Gageheight, feet': 'gageheight_ft',
                                                               'Dis-charge, ft3/s':'Discharge_cfs'})
        
        self.NWIS_sites = self.NWIS_sites[self.NWIS_sites.gageheight_ft != '--']


        self.NWIS_sites = self.NWIS_sites.set_index('StationNumber')
        
        
         # Remove unnecessary site information
        for i in self.NWIS_sites.index:
            if len(str(i)) > 8:
                self.NWIS_sites = self.NWIS_sites.drop(i)

        #remove when confirmed it works
       # NWIS_sites = NWIS_sites[2:3]

        self.site_id = self.NWIS_sites.index

        #set up Pandas DF for state streamstats

        Streamstats_cols = ['NWIS_siteid', 'Drainage_area_mi2', 'Mean_Basin_Elev_ft', 'Perc_Forest', 'Perc_Develop',
                         'Perc_Imperv', 'Perc_Herbace', 'Perc_Slop_30', 'Mean_Ann_Precip_in']

        self.State_NWIS_Stats = pd.DataFrame(columns = Streamstats_cols)
        
        #set counter break to prevent blockage of Public IP address
        #count = 0
        
        print('Calculating NWIS streamflow id characteristics for ', len(self.site_id), ' sites in ', state)

        pbar = ProgressBar()
        for site in pbar(self.site_id):
            
            try:
                siteinfo = self.NWIS_sites['station_name'][site]

                print('Calculating the summary statistics of the catchment for ', siteinfo, ', USGS: ',site)
                NWISinfo = nwis.get_record(sites=site, service='site')

                lat, lon = NWISinfo['dec_lat_va'][0],NWISinfo['dec_long_va'][0]
                ws = streamstats.Watershed(lat=lat, lon=lon)

                NWISindex = ['NWIS_site_id', 'NWIS_sitename', 'Drainage_area_mi2', 'Mean_Basin_Elev_ft', 'Perc_Forest', 'Perc_Develop',
                             'Perc_Imperv', 'Perc_Herbace', 'Perc_Slop_30', 'Mean_Ann_Precip_in', 'Ann_low_cfs', 'Ann_mean_cfs', 'Ann_hi_cfs']


                #get stream statististics
                self.Param="00060"
                StartYr='1970'
                EndYr='2021'

                annual_stats = nwis.get_stats(sites=site,
                                      parameterCd=self.Param,
                                      statReportType='annual',
                                      startDt=StartYr,
                                      endDt=EndYr)

                mean_ann_low = annual_stats[0].nsmallest(1, 'mean_va')
                mean_ann_low = mean_ann_low['mean_va'].values[0]

                mean_ann = np.round(np.mean(annual_stats[0]['mean_va']),0)

                mean_ann_hi = annual_stats[0].nlargest(1, 'mean_va')
                mean_ann_hi = mean_ann_hi['mean_va'].values[0]


                try:
                    darea = ws.get_characteristic('DRNAREA')['value']
                except KeyError:
                    darea = np.nan
                except ValueError:
                    darea = np.nan

                try:
                    elev = ws.get_characteristic('ELEV')['value']
                except KeyError:
                    elev = np.nan
                except ValueError:
                    elev = np.nan

                try:
                    forest = ws.get_characteristic('FOREST')['value']
                except KeyError:
                    forest = np.nan
                except ValueError:
                    forest = np.nan

                try:
                    dev_area = ws.get_characteristic('LC11DEV')['value']
                except KeyError:
                    dev_area = np.nan
                except ValueError:
                    dev_area = np.nan

                try:
                    imp_area = ws.get_characteristic('LC11IMP')['value']
                except KeyError:
                    imp_area = np.nan
                except ValueError:
                    imp_area = np.nan

                try:
                    herb_area = ws.get_characteristic('LU92HRBN')['value']
                except KeyError:
                    herb_area = np.nan
                except ValueError:
                    herb_area = np.nan

                try:
                    perc_slope = ws.get_characteristic('SLOP30_10M')['value']
                except KeyError:
                    perc_slope = np.nan
                except ValueError:
                    perc_slope = np.nan

                try:
                    precip = ws.get_characteristic('PRECIP')['value']
                except KeyError:
                    precip = np.nan
                except ValueError:
                    precip = np.nan


                NWISvalues = [site,siteinfo, darea, elev,forest, dev_area, imp_area, herb_area, perc_slope, precip, mean_ann_low, mean_ann, mean_ann_hi]


                Catchment_Stats = pd.DataFrame(data = NWISvalues, index = NWISindex).T

                self.State_NWIS_Stats = self.State_NWIS_Stats.append(Catchment_Stats)
                
            except:
                time.sleep(181)
                print('Taking three minute break to prevent the blocking of IP Address') 
                
        colorder =[
    'NWIS_site_id',	'NWIS_sitename','Drainage_area_mi2','Mean_Basin_Elev_ft',
    'Perc_Forest', 'Perc_Develop','Perc_Imperv','Perc_Herbace','Perc_Slop_30',
    'Mean_Ann_Precip_in','Ann_low_cfs', 'Ann_mean_cfs','Ann_hi_cfs'
]

        del self.State_NWIS_Stats['NWIS_siteid']

        self.State_NWIS_Stats = self.State_NWIS_Stats[colorder]

        self.State_NWIS_Stats = self.State_NWIS_Stats.reset_index(drop = True)

        self.State_NWIS_Stats.to_csv(self.cwd+'/State_NWIS_StreamStats/StreamStats_'+state+'.csv')

        
        
    def class_eval_state(self, category):
        
        self.category = category
        self.cat_breaks = self.category+'_breaks'
        
        #remove rows with no value for category of interest
        self.df.drop(self.df[self.df[self.category]<0.00001].index, inplace = True)
            

        try: 
            breaks = jenkspy.jenks_breaks(self.df[self.category], n_classes=5)
            print('Categorical breaks for ', self.category, ': ',  breaks)
            self.df[self.cat_breaks] = pd.cut(self.df[self.category],
                                    bins=breaks,
                                    labels=['vsmall', 'small', 'medium', 'large', 'vlarge'],
                                                include_lowest=True)
            self.Catchment_Category()

        except ValueError:
            print('Not enough locations in this dataframe to categorize')    



        self.df = self.df.reset_index(drop = True)



    def Catchment_Category(self):
        #create df for each jenks category
        self.df_vsmall = self.df[self.df[self.cat_breaks]=='vsmall'].reset_index(drop = True)
        self.df_small = self.df[self.df[self.cat_breaks]=='small'].reset_index(drop = True)
        self.df_medium = self.df[self.df[self.cat_breaks]=='medium'].reset_index(drop = True)
        self.df_large= self.df[self.df[self.cat_breaks]=='large'].reset_index(drop = True)
        self.df_vlarge = self.df[self.df[self.cat_breaks]=='vlarge'].reset_index(drop = True)

    def NWIS_retrieve(self, df):
        # Retrieve data from a number of sites
        print('Retrieving USGS sites ', list(df.NWIS_site_id), ' data')
        self.NWIS_sites = list(df.NWIS_site_id)
        
        #self.NWIS_data = pd.DataFrame(columns = self.NWIS_sites)
        pbar = ProgressBar()
        for site in pbar(self.NWIS_sites):
            print('Getting data for: ', site)
            
            try:
                service = IVDataService()
                usgs_data = service.get(
                    sites=str(site),
                    startDT= self.startDT,
                    endDT=self.endDT
                    )

                #Get Daily mean for Model comparision
                usgs_meanflow = pd.DataFrame(usgs_data.reset_index().groupby(pd.Grouper(key = 'value_time', freq = self.freq))['value'].mean())
                usgs_meanflow = usgs_meanflow.reset_index()

                #add key site information
                #make obs data the same as temporal means
                usgs_data = usgs_data.head(len(usgs_meanflow))

                #remove obs streamflow
                del usgs_data['value']
                del usgs_data['value_time']

                #connect mean temporal with other key info
                usgs_meanflow = pd.concat([usgs_meanflow, usgs_data], axis=1)
                usgs_meanflow = usgs_meanflow.rename(columns={'value_time':'Datetime', 'value':'USGS_flow','usgs_site_code':'USGS_ID', 'variable_name':'variable'})
                usgs_meanflow = usgs_meanflow.set_index('Datetime')
                usgs_meanflow.to_hdf(self.cwd+'/Data/NWIS/NWIS_sites_'+state+'.h5', key = site)
                #self.NWIS_data[site] = usgs_meanflow['USGS_flow']
                
            except:
                siteA = '0'+str(site)
                service = IVDataService()
                usgs_data = service.get(
                    sites=siteA,
                    startDT= self.startDT,
                    endDT=self.endDT
                    )

                #Get Daily mean for Model comparision
                usgs_meanflow = pd.DataFrame(usgs_data.reset_index().groupby(pd.Grouper(key = 'value_time', freq = self.freq))['value'].mean())
                usgs_meanflow = usgs_meanflow.reset_index()

                #add key site information
                #make obs data the same as temporal means
                usgs_data = usgs_data.head(len(usgs_meanflow))

                #remove obs streamflow
                del usgs_data['value']
                del usgs_data['value_time']

                #connect mean temporal with other key info
                usgs_meanflow = pd.concat([usgs_meanflow, usgs_data], axis=1)
                usgs_meanflow = usgs_meanflow.rename(columns={'value_time':'Datetime', 'value':'USGS_flow','usgs_site_code':'USGS_ID', 'variable_name':'variable'})
                usgs_meanflow = usgs_meanflow.set_index('Datetime')
                usgs_meanflow.to_hdf(self.cwd+'/Data/NWIS/NWIS_sites_'+state+'.h5', key = site)
                #self.NWIS_data[site] = usgs_meanflow['USGS_flow']
                
                
                
                
    def get_single_NWIS_site(self, site):
        # Retrieve data from a number of sites
        print('Retrieving USGS site: ', site, ' data')
       
        try:
            service = IVDataService()
            usgs_data = service.get(
                sites=str(site),
                startDT= self.startDT,
                endDT=self.endDT
                )

            #Get Daily mean for Model comparision
            usgs_meanflow = pd.DataFrame(usgs_data.reset_index().groupby(pd.Grouper(key = 'value_time', freq = self.freq))['value'].mean())
            usgs_meanflow = usgs_meanflow.reset_index()

            #add key site information
            #make obs data the same as temporal means
            usgs_data = usgs_data.head(len(usgs_meanflow))

            #remove obs streamflow
            del usgs_data['value']
            del usgs_data['value_time']

            #connect mean temporal with other key info
            usgs_meanflow = pd.concat([usgs_meanflow, usgs_data], axis=1)
            usgs_meanflow = usgs_meanflow.rename(columns={'value_time':'Datetime', 'value':'USGS_flow','usgs_site_code':'USGS_ID', 'variable_name':'variable'})
            usgs_meanflow = usgs_meanflow.set_index('Datetime')
            usgs_meanflow.to_hdf(self.cwd+'/Data/NWIS/NWIS_sites_'+self.state+'.h5', key = site)
            #self.NWIS_data[site] = usgs_meanflow['USGS_flow']

        except:
            siteA = '0'+str(site)
            service = IVDataService()
            usgs_data = service.get(
                sites=siteA,
                startDT= self.startDT,
                endDT=self.endDT
                )

            #Get Daily mean for Model comparision
            usgs_meanflow = pd.DataFrame(usgs_data.reset_index().groupby(pd.Grouper(key = 'value_time', freq = self.freq))['value'].mean())
            usgs_meanflow = usgs_meanflow.reset_index()

            #add key site information
            #make obs data the same as temporal means
            usgs_data = usgs_data.head(len(usgs_meanflow))

            #remove obs streamflow
            del usgs_data['value']
            del usgs_data['value_time']

            #connect mean temporal with other key info
            usgs_meanflow = pd.concat([usgs_meanflow, usgs_data], axis=1)
            usgs_meanflow = usgs_meanflow.rename(columns={'value_time':'Datetime', 'value':'USGS_flow','usgs_site_code':'USGS_ID', 'variable_name':'variable'})
            usgs_meanflow = usgs_meanflow.set_index('Datetime')
            usgs_meanflow.to_hdf(self.cwd+'/Data/NWIS/NWIS_sites_'+self.state+'.h5', key = site)
            #self.NWIS_data[site] = usgs_meanflow['USGS_flow']

            
            
    def Model_retrieve(self, df):
        
        # Retrieve data from a number of sites
        print('Retrieving model NHD reaches ', list(df.NHD_reachid), ' data')
        self.comparison_reaches = list(df.NHD_reachid)
        
        pbar = ProgressBar()
        for site in pbar(self.comparison_reaches):
            print('Getting data for: ', site)
            nwm_predictions = data.get_nwm_data(site,  self.startDT,  self.endDT)
            #I think NWM outputs are in cms...
            NHD_meanflow = nwm_predictions.resample(self.freq).mean()*self.cms_to_cfs
            NHD_meanflow = NHD_meanflow.reset_index()
            NHD_meanflow = NHD_meanflow.rename(columns={'time':'Datetime', 'value':'Obs_flow','feature_id':'NHD_segment', 'streamflow':'NHD_flow', 'velocity':'NHD_velocity'})
            NHD_meanflow = NHD_meanflow.set_index('Datetime')
            filepath = self.cwd+'/Data/'+self.model+'/NHD_segments_'+self.state+'.h5',
            NHD_meanflow.to_hdf(filepath, key = site)
           
            
            
            
    def get_single_NWM_reach(self, site):
        
        # Retrieve data from a number of sites
        print('Retrieving NHD Model reach: ', site, ' data')
        nwm_predictions = data.get_nwm_data(site,  self.startDT,  self.endDT)
        #I think NWM outputs are in cms...
        NHD_meanflow = nwm_predictions.resample(self.freq).mean()*self.cms_to_cfs
        NHD_meanflow = NHD_meanflow.reset_index()
        NHD_meanflow = NHD_meanflow.rename(columns={'time':'Datetime', 'value':'Obs_flow','feature_id':'NHD_segment', 'streamflow':'NHD_flow', 'velocity':'NHD_velocity'})
        NHD_meanflow = NHD_meanflow.set_index('Datetime')       
        filepath = self.cwd+'/Data/'+self.model+'/NHD_segments_'+self.state+'.h5',
        NHD_meanflow.to_hdf(filepath, key = site)
            
            
            
            
    def date_range_list(self, start_date, end_date):
        # Return list of datetime.date objects between start_date and end_date (inclusive).
        date_list = []
        curr_date = start_date
        while curr_date <= end_date:
            date_list.append(curr_date)
            curr_date += timedelta(days=1)
        return date_list      

    def prepare_comparison(self, df):
        
        self.comparison_reaches = list(df.NHD_reachid)
        self.NWIS_sites = list(df.NWIS_site_id)
        self.dates = self.date_range_list(pd.to_datetime(self.startDT), pd.to_datetime(self.endDT))
        
        self.NWIS_data = pd.DataFrame(columns = self.NWIS_sites)
        self.Mod_data = pd.DataFrame(columns = self.comparison_reaches)
        
        print('Getting ', self.model, ' data')
        pbar = ProgressBar()
        for site in pbar(self.comparison_reaches):
            filepath = self.cwd+'/Data/'+self.model+'/NHD_segments_'+self.state+'.h5'

            try:

                format = '%Y-%m-%d %H:%M:%S'
                Mod_flow = pd.read_hdf(filepath, key = str(site))
                Mod_flow['time'] ='12:00:00' 
                Mod_flow['Datetime'] = pd.to_datetime(Mod_flow['Datetime']+ ' ' + Mod_flow['time'], format = format)
                Mod_flow.set_index('Datetime', inplace = True)
                Mod_flow = Mod_flow.loc[self.startDT:self.endDT]
                cols = Mod_flow.columns

                flow = self.model + '_flow'


                self.Mod_data[site] = Mod_flow[flow]

            except:
                print('Site: ', site, ' not in database, skipping')
                #remove item from list
                self.comparison_reaches.remove(site)




        #Get NWIS data
        print('Getting NWIS data')
        pbar = ProgressBar()
        for site in pbar(self.NWIS_sites):
            try:
                
                NWIS_meanflow =  pd.read_hdf(self.cwd+'/Data/NWIS/NWIS_sites_'+self.state+'.h5', key = str(site))
                format = '%Y-%m-%d %H:%M:%S'
                NWIS_meanflow.drop_duplicates(subset = 'Datetime', inplace = True)                
                NWIS_meanflow['time'] ='12:00:00' 
                NWIS_meanflow['Datetime'] = pd.to_datetime(NWIS_meanflow['Datetime']+ ' ' + NWIS_meanflow['time'], format = format)
                NWIS_meanflow.set_index('Datetime', inplace = True)
                NWIS_meanflow = NWIS_meanflow.loc[self.startDT:self.endDT]
                self.NWIS_data[site] = np.nan


                #Adjust for different time intervals here
                #Daily
                #if self.freq =='D':
                self.NWIS_data[site] = NWIS_meanflow['USGS_flow']

            except:
                    print('USGS site ', site, ' not in database, skipping')
                    #remove item from list
                    self.NWIS_sites.remove(site)
        

        self.NWIS_column = self.NWIS_data.copy()
        self.NWIS_column = pd.DataFrame(self.NWIS_column.stack(), columns = ['NWIS_flow_cfs'])
        self.NWIS_column = self.NWIS_column.reset_index().drop('level_1',1)

        self.Mod_column = self.Mod_data.copy()
        col = self.model+'_flow_cfs'
        self.Mod_column = pd.DataFrame(self.Mod_column.stack(), columns = [col])
        self.Mod_column = self.Mod_column.reset_index().drop('level_1',1)

        
            
            
    def Model_Eval(self, df, size):

       # self.prepare_comparison(df)
        #Creates a total categorical evaluation comparing model performacne
        print('Creating dataframe of all flow predictions to evaluate')
        self.Evaluation = pd.concat([self.Mod_column,self.NWIS_column], axis = 1)
        self.Evaluation = self.Evaluation.T.drop_duplicates().T     
        self.Evaluation = self.Evaluation.dropna()
        
        num_figs = len(self.comparison_reaches)


        fig, ax = plt.subplots(num_figs ,2, figsize = (10,4*num_figs))

        plot_title = 'Evaluation of ' + self.model + ' predictions related to watershed: ' + self.category + '-'+ size

        fig.suptitle(plot_title, y = 0.89)
        #fig.tight_layout()
        
        
        self.Mod_data['datetime'] = self.dates
        self.Mod_data.set_index('datetime', inplace = True)
        
        self.NWIS_data['datetime'] = self.dates
        self.NWIS_data.set_index('datetime', inplace = True)

        for i in np.arange(0,num_figs,1):
            reach = self.comparison_reaches[i]
            site = self.NWIS_sites[i]

            NWIS_site_lab = 'USGS: ' + str(site)
            Mod_reach_lab = self.model + ': ' + str(reach)

            max_flow = max(max(self.NWIS_data[site]), max(self.Mod_data[reach]))
            min_flow = min(min(self.NWIS_data[site]), min(self.Mod_data[reach]))
            
            plt.subplots_adjust(hspace=0.5)

            ax[i,0].plot(self.Mod_data.index, self.Mod_data[reach], color = 'blue', label = Mod_reach_lab)
            ax[i,0].plot(self.NWIS_data.index, self.NWIS_data[site], color = 'orange',  label = NWIS_site_lab)
            ax[i,0].fill_between(self.NWIS_data.index, self.Mod_data[reach], self.NWIS_data[site], where= self.Mod_data[reach] >= self.NWIS_data[site], facecolor='orange', alpha=0.2, interpolate=True)
            ax[i,0].fill_between(self.NWIS_data.index, self.Mod_data[reach], self.NWIS_data[site], where= self.Mod_data[reach] < self.NWIS_data[site], facecolor='blue', alpha=0.2, interpolate=True)
            ax[i,0].set_xlabel('Datetime')
            ax[i,0].set_ylabel('Discharge (cfs)')
            ax[i,0].tick_params(axis='x', rotation = 45)
            ax[i,0].legend()
           

            ax[i,1].scatter(self.NWIS_data[site], self.Mod_data[reach], color = 'black')
            ax[i,1].plot([min_flow, max_flow],[min_flow, max_flow], ls = '--', c='red')
            ax[i,1].set_xlabel('Observed USGS (cfs)')
            ylab = self.model+ ' Predictions (cfs)'
            ax[i,1].set_ylabel(ylab)

        #calculate some performance metrics
        model_cfs = self.model+'_flow_cfs'
        r2 = r2_score(self.Evaluation.NWIS_flow_cfs, self.Evaluation.model_cfs)
        rmse = mean_squared_error(self.Evaluation.NWIS_flow_cfs, self.Evaluation.model_cfs, squared=False)
        maxerror = max_error(self.Evaluation.NWIS_flow_cfs, self.Evaluation.model_cfs)
        MAPE = mean_absolute_percentage_error(self.Evaluation.NWIS_flow_cfs, self.Evaluation.model_cfs)*100
        kge, r, alpha, beta = he.evaluator(he.kge, self.Evaluation.model_cfs.astype('float32'), self.Evaluation.NWIS_flow_cfs.astype('float32'))

        print('The '+ self.model+ ' demonstrates the following overall performance in catchments exhibiting ', size, ' ', self.category)
        #print('R2 = ', r2)
        print('RMSE = ', rmse, 'cfs')
        print('Maximum error = ', maxerror, 'cfs')
        print('Mean Absolute Percentage Error = ', MAPE, '%')
        print('Kling-Gupta Efficiency = ', kge[0])
        
        
        
        
        
    def Interactive_Model_Eval(self, freq, supply):
        self.freq = freq

        if self.freq == 'D':
            self.units = 'cfs'
        else:
            self.units = 'Acre-Feet'

        #Adjust for different time intervals here
        #Daily
        if self.freq == 'D':
            self.NWIS_data_resampled = self.NWIS_data.copy()
            self.Mod_data_resampled = self.Mod_data.copy()

        #Monthly, Quarterly, Annual
        if self.freq !='D':
            #NWIS
            self.NWIS_data_resampled = self.NWIS_data.copy()*self.cfsday_AFday
            self.NWIS_data_resampled = self.NWIS_data_resampled.resample(self.freq).sum()
            #Modeled
            self.Mod_data_resampled = self.Mod_data.copy()*self.cfsday_AFday
            self.Mod_data_resampled = self.Mod_data_resampled.resample(self.freq).sum()
            
        if supply == True:
            #NWIS
            #Get Columns names
            columns = self.NWIS_data_resampled.columns

            #set up cumulative monthly values
            self.NWIS_data_resampled['Year'] = self.NWIS_data_resampled.index.year

            self.NWIS_CumSum = pd.DataFrame(columns=columns)

            for site in columns:
                self.NWIS_CumSum[site] = self.NWIS_data_resampled.groupby(['Year'])[site].cumsum()

            #Model
            #Get Columns names
            columns = self.Mod_data_resampled.columns

            #set up cumulative monthly values
            self.Mod_data_resampled['Year'] = self.Mod_data_resampled.index.year

            self.Mod_CumSum = pd.DataFrame(columns=columns)

            for site in columns:
                self.Mod_CumSum[site] = self.Mod_data_resampled.groupby(['Year'])[site].cumsum()
                
            #set the Mod and NWIS resampled data == to the CumSum Df's
            self.NWIS_data_resampled = self.NWIS_CumSum
            self.Mod_data_resampled =self.Mod_CumSum

        RMSE = []
        MAXERROR = []
        MAPE = []
        KGE = []

        for row in np.arange(0,len(self.df),1):
            #Get NWIS id
            NWISid = self.df['NWIS_site_id'][row]
            #Get Model reach id
            reachid = 'NHD_reachid'
            modid = self.df[reachid][row]
            #get observed and prediction data
            obs = self.NWIS_data_resampled[NWISid]
            mod = self.Mod_data_resampled[modid]
            
            #remove na values or 0
            df = pd.DataFrame()
            df['obs'] = obs
            df['mod'] = mod.astype('float64')
            df['error'] = df['obs'] - df['mod']
            df['P_error'] = abs(df['error']/df['obs'])*100
            #drop inf values
            df.replace([np.inf, -np.inf], np.nan, inplace = True)
            df.dropna(inplace = True)
            
            obs = df['obs']
            mod = df['mod']

            #calculate scoring
            rmse = round(mean_squared_error(obs, mod, squared=False))
            maxerror = round(max_error(obs, mod))
            mape = df.P_error.mean()
            kge, r, alpha, beta = he.evaluator(he.kge, mod.astype('float32'), obs.astype('float32'))

            RMSE.append(rmse)
            MAXERROR.append(maxerror)
            MAPE.append(mape)
            KGE.append(kge[0])

        #Connect model evaluation to a DF, add in relevant information concerning LULC
        Eval = pd.DataFrame()
        Eval['NWIS_site_id'] = self.df['NWIS_site_id']
        Eval[reachid] = self.df[reachid]
        Eval['Location'] = self.df['NWIS_sitename']
        Eval['RMSE'] = RMSE
        Eval['MaxError'] = MAXERROR
        Eval['MAPE'] = MAPE
        Eval['KGE'] = KGE
        Eval['Drainage_area_mi2'] = self.df['Drainage_area_mi2']
        Eval['Mean_Basin_Elev_ft'] = self.df['Mean_Basin_Elev_ft']
        Eval['Perc_Forest'] = self.df['Perc_Forest']
        Eval['Perc_Imperv'] = self.df['Perc_Imperv']
        Eval['Perc_Herbace'] = self.df['Perc_Herbace']
        Eval['Mean_Ann_Precip_in'] = self.df['Mean_Ann_Precip_in']
        Eval['Ann_low_cfs'] = self.df['Ann_low_cfs']
        Eval['Ann_mean_cfs'] = self.df['Ann_mean_cfs']
        Eval['Ann_hi_cfs'] = self.df['Ann_hi_cfs']
        Eval['Location'] = self.df['NWIS_sitename']
        Eval[self.category] = self.df[self.category]

        #sort dataframe and reindex
        self.Eval = Eval.sort_values('KGE', ascending = False).reset_index(drop = True)    
        #display evaluation DF
        display(self.Eval)
        
        #plot the model performance vs LULC to identify any relationships indicating where/why model performance
        #does well or poor
        #make all very negative KGE values -1
        self.Eval['KGE'][self.Eval['KGE'] < -1] = -1

        fig, ax = plt.subplots(3, 3, figsize = (11,11))
        fig.suptitle('Watershed Charcteristics vs. Model Performance', fontsize = 16)

        ax1 = ['Drainage_area_mi2', 'Mean_Basin_Elev_ft', 'Perc_Forest']
        for var in np.arange(0,len(ax1),1):
            variable = ax1[var]
             #remove na values for variables to make trendline
            cols = ['KGE', variable]
            df = self.Eval[cols]
            df.dropna(axis = 0, inplace = True)
            x = df['KGE']
            y = df[variable]
            ax[0,var].scatter(x = x, y = y)
            ax[0,var].set_ylabel(ax1[var])
             #add trendline
            #calculate equation for trendline
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                #add trendline to plot
                ax[0,var].plot(x, p(x), color = 'r', linestyle = '--')
            except:
                pass

        ax2 = ['Perc_Imperv', 'Perc_Herbace', 'Mean_Ann_Precip_in']
        for var in np.arange(0,len(ax2),1):
            variable = ax2[var]
             #remove na values for variables to make trendline
            cols = ['KGE', variable]
            df = self.Eval[cols]
            df.dropna(axis = 0, inplace = True)
            x = df['KGE']
            y = df[variable]
            ax[1,var].scatter(x = x, y = y)
            ax[1,var].set_ylabel(ax2[var])
            #add trendline
            #calculate equation for trendline
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                #add trendline to plot
                ax[1,var].plot(x, p(x), color = 'r', linestyle = '--')
            except:
                pass

        ax3 = ['Ann_low_cfs', 'Ann_mean_cfs', 'Ann_hi_cfs']
        for var in np.arange(0,len(ax3),1):
            variable = ax3[var]
            #remove na values for variables to make trendline
            cols = ['KGE', variable]
            df = self.Eval[cols]
            df.dropna(axis = 0, inplace = True)
            x = df['KGE']
            y = df[variable]
            ax[2,var].scatter(x = x, y = y)
            ax[2,var].set_xlabel('Model Performance (KGE)')
            ax[2,var].set_ylabel(ax3[var])
             #add trendline
            #calculate equation for trendline
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                #add trendline to plot
                ax[2,var].plot(x, p(x), color = 'r', linestyle = '--')
            except:
                pass


        plt.tight_layout()
        plt.show()
        
        num_figs = len(self.Eval)
        for i in np.arange(0,num_figs,1):
            

            reach = self.Eval[reachid][i]
            site = self.Eval['NWIS_site_id'][i]
            #print(site, reach)
 
            sitename = self.Eval.Location[i]
            sitestat = str(self.Eval[self.category][i])

            plot_title = 'Performance of ' + self.model +' predictions related to: ' + self.category +  '\n' + sitename + '\n'+ self.category +': ' + sitestat + ', classified as: '+ self.size


            NWIS_site_lab = 'USGS: ' + str(site)
            Mod_reach_lab = self.model + ': NHD ' + str(reach)

            Eval_cols = [NWIS_site_lab, Mod_reach_lab]

            #Adjust for different time intervals here
            #Daily

            Eval_df = pd.DataFrame(index = self.NWIS_data_resampled.index, columns = Eval_cols)
            Eval_df[Mod_reach_lab] = self.Mod_data_resampled[reach]
            Eval_df[NWIS_site_lab] = self.NWIS_data_resampled[site]


            Eval_df = Eval_df.dropna()

            if Eval_df.shape[0] > 0:

                #need to have datetime fixed
                Eval_df = Eval_df.reset_index()
                Eval_df['Datetime'] = pd.to_datetime(Eval_df['Datetime'])
                Eval_df.set_index('Datetime', inplace = True, drop = True)
                
                #get observed and prediction data
                obs = Eval_df[NWIS_site_lab]
                mod = Eval_df[Mod_reach_lab]

                #remove na values or 0
                df = pd.DataFrame()
                df['obs'] = obs
                df['mod'] = mod.astype('float64')
                df['error'] = df['obs'] - df['mod']
                df['P_error'] = abs(df['error']/df['obs'])*100
                #drop inf values
                df.replace([np.inf, -np.inf], np.nan, inplace = True)
                df.dropna(inplace = True)

                obs = df['obs']
                mod = df['mod']

                #calculate scoring
                rmse = round(mean_squared_error(obs, mod, squared=False))
                maxerror = round(max_error(obs, mod))
                MAPE = round(mean_absolute_percentage_error(obs, mod)*100)
                kge, r, alpha, beta = he.evaluator(he.kge, mod.astype('float32'), obs.astype('float32'))
                
                #set limit to MAPE error
                if MAPE > 1000:
                    MAPE ='> 1000'

                rmse_phrase = 'RMSE: ' + str(rmse) + ' ' + self.units
                error_phrase = 'Max Error: ' + str(maxerror) + ' ' + self.units
                mape_phrase = 'MAPE: ' + str(MAPE) + '%'
                kge_phrase = 'kge: ' + str(round(kge[0],2))

                max_flow = max(max(Eval_df[NWIS_site_lab]), max(Eval_df[Mod_reach_lab]))
                min_flow = min(min(Eval_df[NWIS_site_lab]), min(Eval_df[Mod_reach_lab]))

                flow_range = np.arange(min_flow, max_flow, (max_flow-min_flow)/100)
                
                if self.freq == 'A':
                    bbox_L = -int(round((len(Eval_df)*.32),0))
                    text_bbox_L = -int(round((len(Eval_df)*.22),0))
                    
                else:
                    bbox_L = -int(round((len(Eval_df)*.32),0))
                    text_bbox_L = -int(round((len(Eval_df)*.16),0))

                Discharge_lab = 'Discharge (' +self.units +')'
                Obs_Discharge_lab = ' Observed Discharge (' +self.units +')'
                Mod_Discharge_lab = self.model +' Discharge (' +self.units +')'


                NWIS_hydrograph = hv.Curve((Eval_df.index, Eval_df[NWIS_site_lab]), 'DateTime', Discharge_lab, label = NWIS_site_lab).opts(title = plot_title, tools = ['hover'], color = 'orange')
                Mod_hydrograph = hv.Curve((Eval_df.index, Eval_df[Mod_reach_lab]), 'DateTime', Discharge_lab, label = Mod_reach_lab).opts(tools = ['hover'], color = 'blue')
                RMSE_hv = hv.Text(Eval_df.index[text_bbox_L],max_flow*.93, rmse_phrase, fontsize = 8)
                Error_hv = hv.Text(Eval_df.index[text_bbox_L],max_flow*.83, error_phrase, fontsize = 8)
                MAPE_hv = hv.Text(Eval_df.index[text_bbox_L],max_flow*.73, mape_phrase, fontsize = 8)
                KGE_hv = hv.Text(Eval_df.index[text_bbox_L],max_flow*.63, kge_phrase, fontsize = 8)
                textbox_hv = hv.Rectangles([(Eval_df.index[bbox_L], max_flow*.56, Eval_df.index[-1], max_flow*.99)]).opts(color = 'white')

                Mod_NWIS_Scatter = hv.Scatter((Eval_df[NWIS_site_lab], Eval_df[Mod_reach_lab]), Obs_Discharge_lab, Mod_Discharge_lab).opts(tools = ['hover'], color = 'blue', xrotation=45)
                Mod_NWIS_one2one = hv.Curve((flow_range, flow_range)).opts(color = 'red', line_dash='dashed')


                #display((NWIS_hydrograph * Mod_hydrograph).opts(width=600, legend_position='top_left', tools=['hover']) + (Mod_NWIS_Scatter*Mod_NWIS_one2one).opts(shared_axes = False))
                display((NWIS_hydrograph * Mod_hydrograph*textbox_hv*RMSE_hv*Error_hv*MAPE_hv*KGE_hv).opts(width=600, legend_position='top_left', tools=['hover']) + (Mod_NWIS_Scatter*Mod_NWIS_one2one).opts(shared_axes = False))

            else:
                print('No data for NWIS site: ', str(NWIS_site_lab), ' skipping.')
    
            
    #streamstats does not get lat long, we need this to do any NWIS geospatial work
    #https://github.com/hyriver/HyRiver-examples/blob/main/notebooks/dam_impact.ipynb
    def more_StreamStats(self, state, cwd):
        start = "2019-01-01"
        end = "2020-01-01"
        nwis = NWIS()
        query = {
            "stateCd": state,
            "startDt": start,
            "endDt": end,
            "outputDataTypeCd": "dv",  # daily values
            "hasDataTypeCd": "dv",  # daily values
            "parameterCd": "00060",  # discharge
        }
        sites = nwis.get_info(query)
        sites = sites.drop_duplicates(subset = ['site_no'])
        sites['site_no'] = sites['site_no'].astype(str).astype('int64')
        sites = sites[sites['site_no'] < 20000000].reset_index(drop =  True)
        sites['site_no'] = sites['site_no'].astype(str)

        for site in np.arange(0, len(sites),1):
            if len(sites['site_no'][site]) == 7:
                sites['site_no'][site] = '0'+sites['site_no'][site]


        cols = ['site_no', 'station_nm', 'dec_lat_va',
               'dec_long_va', 'alt_va',
               'alt_acy_va', 'huc_cd', 'parm_cd',
               'begin_date', 'end_date',
               'drain_sqkm',  'geometry']
        sites = sites[cols]    

        sites.to_csv(cwd+ '/Data/StreamStats/more_stats/'+ state+'.csv')
        
        
        
        #Map locations and scoring of sites
    def Map_Plot_Eval(self, freq, df, size, supply):
        self.freq = freq
        self.df = df
        self.size = size

        if self.freq == 'D':
            self.units = 'cfs'
        else:
            self.units = 'Acre-Feet'

        yaxis = 'Streamflow (' + self.units +')'
        
        #Get data and prepare
        self.prepare_comparison(self.df)

        #Adjust for different time intervals here
        #Daily
        if self.freq == 'D':
            self.NWIS_data_resampled = self.NWIS_data.copy()
            self.Mod_data_resampled = self.Mod_data.copy()


        #Monthly, Quarterly, Annual
        if self.freq !='D':
            #NWIS
            self.NWIS_data_resampled = self.NWIS_data.copy()*self.cfsday_AFday
            self.NWIS_data_resampled = self.NWIS_data_resampled.resample(self.freq).sum()
            #Modeled
            self.Mod_data_resampled = self.Mod_data.copy()*self.cfsday_AFday
            self.Mod_data_resampled = self.Mod_data_resampled.resample(self.freq).sum()
            
        if supply == True:
            #NWIS
            #Get Columns names
            columns = self.NWIS_data_resampled.columns

            #set up cumulative monthly values
            self.NWIS_data_resampled['Year'] = self.NWIS_data_resampled.index.year

            self.NWIS_CumSum = pd.DataFrame(columns=columns)

            for site in columns:
                self.NWIS_CumSum[site] = self.NWIS_data_resampled.groupby(['Year'])[site].cumsum()

            #Model
            #Get Columns names
            columns = self.Mod_data_resampled.columns

            #set up cumulative monthly values
            self.Mod_data_resampled['Year'] = self.Mod_data_resampled.index.year

            self.Mod_CumSum = pd.DataFrame(columns=columns)

            for site in columns:
                self.Mod_CumSum[site] = self.Mod_data_resampled.groupby(['Year'])[site].cumsum()
                
            #set the Mod and NWIS resampled data == to the CumSum Df's
            self.NWIS_data_resampled = self.NWIS_CumSum
            self.Mod_data_resampled =self.Mod_CumSum

        print('Plotting monitoring station locations')
        cols =  ['NWIS_site_id', 'NWIS_sitename', 'NHD_reachid', 'dec_lat_va', 'dec_long_va', 'geometry']

        self.df_map = self.df[cols]
        self.df_map.reset_index(inplace = True, drop = True) 
        #Get Centroid of watershed
        centeroid = self.df_map.dissolve().centroid

        # Create a Map instance
        m = folium.Map(location=[centeroid.y[0], centeroid.x[0]], tiles = 'Stamen Terrain', zoom_start=8, 
                       control_scale=True)
        #add legend to map
        colormap = cm.StepColormap(colors = ['darkred', 'r', 'orange', 'g'], vmin = -1, vmax = 1, index = [-1,-0.4,0,0.3,1])
        colormap.caption = 'Model Performance (KGE)'
        m.add_child(colormap)

        ax = AxisProperties(
        labels=PropertySet(
            angle=ValueRef(value=300),
            align=ValueRef(value='right')
                )
            )

        for i in np.arange(0, len(self.df_map),1):
            #get site information
            site = self.df_map['NWIS_site_id'][i]
            USGSsite = 'USGS station id: ' + site
            site_name = self.df_map['NWIS_sitename'][i]

            reach = self.df_map['NHD_reachid'][i]
            Modreach = self.model +' reach id: ' + str(reach)
            
            #get modeled and observed information for each site
            df = pd.DataFrame(self.NWIS_data_resampled[site])
            df = df.rename(columns = {site: USGSsite})
            df[Modreach] = pd.DataFrame(self.Mod_data_resampled[reach])
            
            #remove na values or 0, this evaluates the model only on NWIS observations
            df_narem = pd.DataFrame()
            df_narem['obs'] = self.NWIS_data_resampled[site].astype('float64')
            df_narem['mod'] = self.Mod_data_resampled[reach].astype('float64')
            df_narem['error'] = df_narem['obs'] - df_narem['mod']
            df_narem['P_error'] = abs(df_narem['error']/df_narem['obs'])*100
            #drop inf values
            df_narem.replace([np.inf, -np.inf], np.nan, inplace = True)
            df_narem.dropna(inplace = True)

            obs = df_narem['obs']
            mod = df_narem['mod']

            #calculate scoring
            kge, r, alpha, beta = he.evaluator(he.kge, mod.astype('float32'), obs.astype('float32'))
   
            #get modeled and observed information for each site
            #df = pd.DataFrame(self.NWIS_data_resampled[site])
            #df = df.rename(columns = {site: USGSsite})
            #df[Modreach] = pd.DataFrame(self.Mod_data_resampled[reach])
            
            #set the color of marker by model performance
            #kge, r, alpha, beta = he.evaluator(he.kge, df[Modreach].astype('float32'), df[USGSsite].astype('float32'))

            if kge[0] > 0.30:
                color = 'green'

            elif kge[0] > 0.0:
                color = 'orange'

            elif kge[0] > -0.40:
                color = 'red'

            else:
                color = 'darkredred'
            
            
            title_size = 14

            #create graph and convert to json
            graph = vincent.Line(df, height=300, width=500)
            graph.axis_titles(x='Datetime', y=yaxis)
            graph.legend(title= site_name)
            graph.colors(brew='Set1')
            #graph.axes[0].properties = ax
            graph.x_axis_properties(title_size=title_size, title_offset=35,
                          label_angle=300, label_align='right', color=None)
            graph.y_axis_properties(title_size=title_size, title_offset=-30,
                          label_angle=None, label_align='right', color=None)

            data = json.loads(graph.to_json())

            #Add marker with point to map, https://fontawesome.com/v4/cheatsheet/
            lat = self.df_map['dec_lat_va'][i]
            long = self.df_map['dec_long_va'][i]
            mk = features.Marker([lat, long], icon=folium.Icon(color=color, icon = 'fa-navicon', prefix = 'fa'))
            p = folium.Popup("Hello")
            v = features.Vega(data, width="100%", height="100%")

            mk.add_child(p)
            p.add_child(v)
            m.add_child(mk)


        display(m)

     
    
class HUC_Eval():
    def __init__(self, model , HUCid, startDT, endDT, cwd):
        self = self
        #self.df =df
        self.startDT = startDT
        self.endDT = endDT
        self.cwd = cwd
        self.cms_to_cfs = 35.314666212661
        self.model = model
        self.HUCid = HUCid
        self.cfsday_AFday = 1.983
        self.freqkeys = {
                        'D': 'Daily',
                        'M': 'Monthly',
                        'Q': 'Quarterly',
                        'A': 'Annual'
                        }


    def date_range_list(self):
        # Return list of datetime.date objects between start_date and end_date (inclusive).
        date_list = []
        curr_date = pd.to_datetime(self.startDT)
        while curr_date <= pd.to_datetime(self.endDT):
            date_list.append(curr_date)
            curr_date += timedelta(days=1)
        self.dates = date_list

    '''
     Function for getting state id from lat long, needed to get NWIS and NHD streamflow information
     '''   

    def Lat_Long_to_state(self, row):
        coord = f"{row['dec_lat_va']}, {row['dec_long_va']}"
        location = geolocator.reverse(coord, exactly_one=True)
        address = location.raw['address']
        state = address.get('state', '')
        row['state'] = state
        #row['state_id'] = row['state'].map(state_code_map)
        return row

    '''
    Get WBD HUC data, how to add in multiple hucs at once from same HU?
    '''
    def Join_WBD_StreamStats(self):
        print('Getting geospatial information for HUC: ', self.HUCid)
        try:
            #Get HUC level
            self.HUC_length = 'huc'+str(len(self.HUCid[0]))

            #columns to keep
            self.HUC_cols = ['areaacres', 'areasqkm', 'states', self.HUC_length, 'name', 'shape_Length', 'shape_Area', 'geometry']
            self.HUC_Geo = gpd.GeoDataFrame(columns = self.HUC_cols, geometry = 'geometry')
            print(self.HUCid)
            
            t0 = time.time()
            for h in self.HUCid:
                HU = h[:2]
                HUCunit = 'WBDHU'+str(len(h))
                gdb_file = self.cwd+'/Data/WBD/WBD_' +HU+'_HU2_GDB/WBD_'+HU+'_HU2_GDB.gdb'

                # Get HUC unit from the .gdb file 
                #load the HUC geopandas df
                HUC_G = gpd.read_file(gdb_file,layer=HUCunit)

                #select HUC
                HUC_G = HUC_G[HUC_G[self.HUC_length] == h] 
                HUC_G = HUC_G[self.HUC_cols]
                self.HUC_Geo = self.HUC_Geo.append(HUC_G)
            t1 = time.time()
            #print('HUC loading took ', t1-t0, 'seconds')

            #Load streamstats and covert to geodataframe
            Streamstats = pd.read_hdf(self.cwd+'/Data/StreamStats/StreamStats3.h5', 'streamstats')
            Streamstats.drop_duplicates(subset = 'NWIS_site_id', inplace = True)

            #Convert to geodataframe
            self.StreamStats = gpd.GeoDataFrame(Streamstats, geometry=gpd.points_from_xy(Streamstats.dec_long_va, Streamstats.dec_lat_va))

            print('Finding NWIS monitoring stations within ', self.HUCid, ' watershed boundary')
            # Join StreamStats with HUC
            self.HUC_NWIS = self.StreamStats.sjoin(self.HUC_Geo, how = 'inner', predicate = 'intersects')
            
            #Somehow duplicate rows occuring, fix added
            self.HUC_NWIS =  self.HUC_NWIS.drop_duplicates()
            print('Creating dataframe of NWIS stations within ', self.HUCid, ' watershed boundary')
            #takes rows with site name
            self.HUC_NWIS = self.HUC_NWIS[self.HUC_NWIS['NWIS_sitename'].notna()] 

        except KeyError:
            print('No monitoring stations in this HUC')
            
            
        if len(self.HUC_NWIS) == 0:
            print('No monitoring stations in this HUC')
    

    def get_NHD_Model_info(self):   
        print('Getting collocated ',  self.model, ' NHD reaches with NWIS monitoring locations')
       #Get NHD reach colocated with NWIS       
        NHD_reaches = []

        for site in self.HUC_NWIS.NWIS_site_id:
            try:
                NHD_NWIS_df = utils.crosswalk(usgs_site_codes=site)
                NHD_segment = NHD_NWIS_df.nwm_feature_id.values[0]
                NHD_reaches.append(NHD_segment)

            except:
                NHD_segment = np.nan
                NHD_reaches.append(NHD_segment)

        self.HUC_NWIS['NHD_reachid'] = NHD_reaches

        self.HUC_NWIS = self.HUC_NWIS.fillna(0.0)

        self.HUC_NWIS['NHD_reachid'] = self.HUC_NWIS['NHD_reachid'].astype(int)

        self.HUC_NWIS = self.HUC_NWIS[self.HUC_NWIS.NHD_reachid != 0]



    def prepare_comparison(self):

        #prepare the daterange
        self.date_range_list()

        self.comparison_reaches = list(self.HUC_NWIS.NHD_reachid)
        self.NWIS_sites = list(self.HUC_NWIS.NWIS_site_id)


        self.NWIS_data = pd.DataFrame(columns = self.NWIS_sites)
        self.Mod_data = pd.DataFrame(columns = self.comparison_reaches)

        self.HUC_NWIS.state_id = self.HUC_NWIS.state_id.str.lower()
        #create a key/dict of site/state id
        NWIS_state_key =  dict(zip(self.HUC_NWIS.NWIS_site_id, 
                                  self.HUC_NWIS.state_id))                           


        Mod_state_key =  dict(zip(self.HUC_NWIS.NHD_reachid, 
                              self.HUC_NWIS.state_id))

        print('Getting ', self.model, ' data')
        pbar = ProgressBar()
        for site in pbar(self.comparison_reaches):
            state = Mod_state_key[site]
            filepath = self.cwd+'/Data/'+self.model+'/NHD_segments_'+state+'.h5'
            try:

                format = '%Y-%m-%d %H:%M:%S'
                Mod_flow = pd.read_hdf(filepath, key = str(site))
                Mod_flow['time'] ='12:00:00' 
                Mod_flow['Datetime'] = pd.to_datetime(Mod_flow['Datetime']+ ' ' + Mod_flow['time'], format = format)
                Mod_flow.set_index('Datetime', inplace = True)
                Mod_flow = Mod_flow.loc[self.startDT:self.endDT]

                flow = self.model + '_flow'


                self.Mod_data[site] = Mod_flow[flow]

            except:
                print(self.model,' site ', site, ' not in database, skipping')
                #remove item from list

        #reset comparison reaches
        self.Mod_data.dropna(axis = 1, inplace = True)
        self.comparison_reaches = self.Mod_data.columns




        #Get NWIS data
        print('Getting NWIS data')
        pbar = ProgressBar()
      
        for site in pbar(self.NWIS_sites):

            try:
                state = NWIS_state_key[site]
                NWIS_meanflow =  pd.read_hdf(self.cwd+'/Data/NWIS/NWIS_sites_'+state+'.h5', key = str(site))
                format = '%Y-%m-%d %H:%M:%S'
                NWIS_meanflow.drop_duplicates(subset = 'Datetime', inplace = True)                
                NWIS_meanflow['time'] ='12:00:00' 
                NWIS_meanflow['Datetime'] = pd.to_datetime(NWIS_meanflow['Datetime']+ ' ' + NWIS_meanflow['time'], format = format)
                NWIS_meanflow.set_index('Datetime', inplace = True)
                NWIS_meanflow = NWIS_meanflow.loc[self.startDT:self.endDT]
                self.NWIS_data[site] = np.nan
                #print(NWIS_meanflow['USGS_flow'][:1])
                self.NWIS_data[site] = NWIS_meanflow['USGS_flow']

            except:
                    print('USGS site ', site, ' not in database, skipping')
                    #remove item from list
                    self.HUC_NWIS = self.HUC_NWIS[self.HUC_NWIS['NWIS_site_id'] !=  site]
        #reset NWIS sites  
        self.HUC_NWIS.reset_index(drop = True, inplace = True)
        self.NWIS_sites = self.HUC_NWIS['NWIS_site_id']

        #need to get the date range of NWIS and adjust modeled flow
        NWIS_dates = self.NWIS_data.index
        self.Mod_data = self.Mod_data.loc[NWIS_dates[0]:NWIS_dates[-1]]

        self.NWIS_column = self.NWIS_data.copy()
        self.NWIS_column = pd.DataFrame(self.NWIS_column.stack(), columns = ['NWIS_flow_cfs'])
        self.NWIS_column = self.NWIS_column.reset_index().drop('level_1',1)

        self.Mod_column = self.Mod_data.copy()
        col = self.model+'_flow_cfs'
        self.Mod_column = pd.DataFrame(self.Mod_column.stack(), columns = [col])
        self.Mod_column = self.Mod_column.reset_index().drop('level_1',1)

    def Interactive_Model_Eval(self, freq, supply):
        self.freq = freq

        if self.freq == 'D':
            self.units = 'cfs'
        else:
            self.units = 'Acre-Feet'

        #Adjust for different time intervals here
        #Daily
        if self.freq == 'D':
            self.NWIS_data_resampled = self.NWIS_data.copy()
            self.Mod_data_resampled = self.Mod_data.copy()

        #Monthly, Quarterly, Annual
        if self.freq !='D':
            #NWIS
            self.NWIS_data_resampled = self.NWIS_data.copy()*self.cfsday_AFday
            self.NWIS_data_resampled = self.NWIS_data_resampled.resample(self.freq).sum()
            #Modeled
            self.Mod_data_resampled = self.Mod_data.copy()*self.cfsday_AFday
            self.Mod_data_resampled = self.Mod_data_resampled.resample(self.freq).sum()
            
        if supply == True:
            #NWIS
            #Get Columns names
            columns = self.NWIS_data_resampled.columns

            #set up cumulative monthly values
            self.NWIS_data_resampled['Year'] = self.NWIS_data_resampled.index.year

            self.NWIS_CumSum = pd.DataFrame(columns=columns)

            for site in columns:
                self.NWIS_CumSum[site] = self.NWIS_data_resampled.groupby(['Year'])[site].cumsum()

            #Model
            #Get Columns names
            columns = self.Mod_data_resampled.columns

            #set up cumulative monthly values
            self.Mod_data_resampled['Year'] = self.Mod_data_resampled.index.year

            self.Mod_CumSum = pd.DataFrame(columns=columns)

            for site in columns:
                self.Mod_CumSum[site] = self.Mod_data_resampled.groupby(['Year'])[site].cumsum()
                
            #set the Mod and NWIS resampled data == to the CumSum Df's
            self.NWIS_data_resampled = self.NWIS_CumSum
            self.Mod_data_resampled =self.Mod_CumSum
            
            
            
        RMSE = []
        MAXERROR = []
        MAPE = []
        KGE = []

        for row in np.arange(0,len(self.HUC_NWIS),1):
            #Get NWIS id
            NWISid = self.HUC_NWIS['NWIS_site_id'][row]
            #Get Model reach id
            reachid = 'NHD_reachid'
            modid = self.HUC_NWIS[reachid][row]
            #get observed and prediction data
            obs = self.NWIS_data_resampled[NWISid]
            mod = self.Mod_data_resampled[modid]
            
            #remove na values or 0
            df = pd.DataFrame()
            df['obs'] = obs
            df['mod'] = mod.astype('float64')
            df['error'] = df['obs'] - df['mod']
            df['P_error'] = abs(df['error']/df['obs'])*100
            #drop inf values
            df.replace([np.inf, -np.inf], np.nan, inplace = True)
            df.dropna(inplace = True)
            
            obs = df['obs']
            mod = df['mod']

            #calculate scoring
            rmse = round(mean_squared_error(obs, mod, squared=False))
            maxerror = round(max_error(obs, mod))
            mape = df.P_error.mean()
            kge, r, alpha, beta = he.evaluator(he.kge, mod.astype('float32'), obs.astype('float32'))

            RMSE.append(rmse)
            MAXERROR.append(maxerror)
            MAPE.append(mape)
            KGE.append(kge[0])

        #Connect model evaluation to a DF, add in relevant information concerning LULC
        Eval = pd.DataFrame()
        Eval['NWIS_site_id'] = self.HUC_NWIS['NWIS_site_id']
        Eval[reachid] = self.HUC_NWIS[reachid]
        Eval['Location'] = self.HUC_NWIS['NWIS_sitename']
        Eval['RMSE'] = RMSE
        Eval['MaxError'] = MAXERROR
        Eval['MAPE'] = MAPE
        Eval['KGE'] = KGE
        Eval['Drainage_area_mi2'] = self.HUC_NWIS['Drainage_area_mi2']
        Eval['Mean_Basin_Elev_ft'] = self.HUC_NWIS['Mean_Basin_Elev_ft']
        Eval['Perc_Forest'] = self.HUC_NWIS['Perc_Forest']
        Eval['Perc_Imperv'] = self.HUC_NWIS['Perc_Imperv']
        Eval['Perc_Herbace'] = self.HUC_NWIS['Perc_Herbace']
        Eval['Mean_Ann_Precip_in'] = self.HUC_NWIS['Mean_Ann_Precip_in']
        Eval['Ann_low_cfs'] = self.HUC_NWIS['Ann_low_cfs']
        Eval['Ann_mean_cfs'] = self.HUC_NWIS['Ann_mean_cfs']
        Eval['Ann_hi_cfs'] = self.HUC_NWIS['Ann_hi_cfs']
        Eval['name'] = self.HUC_NWIS['name']
        Eval[self.HUC_length] = self.HUC_NWIS[self.HUC_length]

        #sort dataframe and reindex
        self.Eval = Eval.sort_values('KGE', ascending = False).reset_index(drop = True)    
        #display evaluation DF
        display(self.Eval)
        
        #plot the model performance vs LULC to identify any relationships indicating where/why model performance
        #does well or poor
        #make all very negative KGE values -1
        self.Eval['KGE'][self.Eval['KGE'] < -1] = -1

        fig, ax = plt.subplots(3, 3, figsize = (11,11))
        fig.suptitle('Watershed Charcteristics vs. Model Performance', fontsize = 16)

        ax1 = ['Drainage_area_mi2', 'Mean_Basin_Elev_ft', 'Perc_Forest']
        for var in np.arange(0,len(ax1),1):
            variable = ax1[var]
            #remove na values for variables to make trendline
            cols = ['KGE', variable]
            df = self.Eval[cols]
            df.dropna(axis = 0, inplace = True)
            x = df['KGE']
            y = df[variable]
            ax[0,var].scatter(x = x, y = y)
            ax[0,var].set_ylabel(ax1[var])
             #add trendline
            #calculate equation for trendline
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                #add trendline to plot
                ax[0,var].plot(x, p(x), color = 'r', linestyle = '--')
            except:
                pass

        ax2 = ['Perc_Imperv', 'Perc_Herbace', 'Mean_Ann_Precip_in']
        for var in np.arange(0,len(ax2),1):
            variable = ax2[var]
             #remove na values for variables to make trendline
            cols = ['KGE', variable]
            df = self.Eval[cols]
            df.dropna(axis = 0, inplace = True)
            x = df['KGE']
            y = df[variable]
            ax[1,var].scatter(x = x, y = y)
            ax[1,var].set_ylabel(ax2[var])
            #add trendline
            #calculate equation for trendline
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                #add trendline to plot
                ax[1,var].plot(x, p(x), color = 'r', linestyle = '--')
            except:
                pass

        ax3 = ['Ann_low_cfs', 'Ann_mean_cfs', 'Ann_hi_cfs']
        for var in np.arange(0,len(ax3),1):
            variable = ax3[var]
             #remove na values for variables to make trendline
            cols = ['KGE', variable]
            df = self.Eval[cols]
            df.dropna(axis = 0, inplace = True)
            x = df['KGE']
            y = df[variable]
            ax[2,var].scatter(x = x, y = y)
            ax[2,var].set_xlabel('Model Performance (KGE)')
            ax[2,var].set_ylabel(ax3[var])
             #add trendline
            #calculate equation for trendline
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                #add trendline to plot
                ax[2,var].plot(x, p(x), color = 'r', linestyle = '--')
            except:
                pass


        plt.tight_layout()
        plt.show()
        
        num_figs = len(self.Eval)
        self.HUC_NWIS.reset_index(inplace = True, drop = True)
        for i in np.arange(0,num_figs,1):

            reach = self.Eval[reachid][i]
            site = self.Eval['NWIS_site_id'][i]
            #print(site, reach)
            sitename = self.Eval.Location[i]
            #sitestat = str(df[self.category][i])

            plot_title = self.Eval['name'][i] +' Basin: HUC' + self.Eval[self.HUC_length][i] + ' ' + self.freqkeys[self.freq]+ ' (' + self.units +') \n Performance of ' + self.model +' predictions, reach: ' + str(reach) + '\n USGS:' + str(site) +' ' + str(sitename)
            #plot_title = self.HUC_NWIS['name'][0] +' Basin' + ' ' + self.freqkeys[self.freq]+ ' (' + self.units +') \n Performance of ' + self.model +' predictions, reach: ' + str(reach) + '\n USGS:' + str(site) +' ' + str(sitename)


            NWIS_site_lab = 'USGS: ' + str(site)
            Mod_reach_lab = self.model + ': NHD ' + str(reach)

            Eval_cols = [NWIS_site_lab, Mod_reach_lab]

            #Adjust for different time intervals here
            #Daily

            Eval_df = pd.DataFrame(index = self.NWIS_data_resampled.index, columns = Eval_cols)
            Eval_df[Mod_reach_lab] = self.Mod_data_resampled[reach]
            Eval_df[NWIS_site_lab] = self.NWIS_data_resampled[site]


            Eval_df = Eval_df.dropna()

            if Eval_df.shape[0] > 0:

                #need to have datetime fixed
                Eval_df = Eval_df.reset_index()
                Eval_df['Datetime'] = pd.to_datetime(Eval_df['Datetime'])
                Eval_df.set_index('Datetime', inplace = True, drop = True)
                
                #get observed and prediction data
                obs = Eval_df[NWIS_site_lab]
                mod = Eval_df[Mod_reach_lab]

                #remove na values or 0
                df = pd.DataFrame()
                df['obs'] = obs
                df['mod'] = mod.astype('float64')
                df['error'] = df['obs'] - df['mod']
                df['P_error'] = abs(df['error']/df['obs'])*100
                #drop inf values
                df.replace([np.inf, -np.inf], np.nan, inplace = True)
                df.dropna(inplace = True)

                obs = df['obs']
                mod = df['mod']

                #calculate scoring
                rmse = round(mean_squared_error(obs, mod, squared=False))
                maxerror = round(max_error(obs, mod))
                MAPE = round(mean_absolute_percentage_error(obs, mod)*100)
                kge, r, alpha, beta = he.evaluator(he.kge, mod.astype('float32'), obs.astype('float32'))
                
                #set limit to MAPE error
                if MAPE > 1000:
                    MAPE ='> 1000'

                rmse_phrase = 'RMSE: ' + str(rmse) + ' ' +  self.units
                error_phrase = 'Max Error: ' + str(maxerror) + ' ' + self.units
                mape_phrase = 'MAPE: ' + str(MAPE) + '%'
                kge_phrase = 'kge: ' + str(round(kge[0],2))
                

                max_flow = max(max(Eval_df[NWIS_site_lab]), max(Eval_df[Mod_reach_lab]))
                min_flow = min(min(Eval_df[NWIS_site_lab]), min(Eval_df[Mod_reach_lab]))

                flow_range = np.arange(min_flow, max_flow, (max_flow-min_flow)/100)

                if self.freq == 'A':
                    bbox_L = -int(round((len(Eval_df)*.32),0))
                    text_bbox_L = -int(round((len(Eval_df)*.18),0))
                    
                else:
                    bbox_L = -int(round((len(Eval_df)*.32),0))
                    text_bbox_L = -int(round((len(Eval_df)*.16),0))

                Discharge_lab = 'Discharge (' +self.units +')'
                Obs_Discharge_lab = ' Observed Discharge (' +self.units +')'
                Mod_Discharge_lab = self.model +' Discharge (' +self.units +')'


                NWIS_hydrograph = hv.Curve((Eval_df.index, Eval_df[NWIS_site_lab]), 'DateTime', Discharge_lab, label = NWIS_site_lab).opts(title = plot_title, tools = ['hover'], color = 'orange')
                Mod_hydrograph = hv.Curve((Eval_df.index, Eval_df[Mod_reach_lab]), 'DateTime', Discharge_lab, label = Mod_reach_lab).opts(tools = ['hover'], color = 'blue')
                RMSE_hv = hv.Text(Eval_df.index[text_bbox_L],max_flow*.93, rmse_phrase, fontsize = 8)
                Error_hv = hv.Text(Eval_df.index[text_bbox_L],max_flow*.83, error_phrase, fontsize = 8)
                MAPE_hv = hv.Text(Eval_df.index[text_bbox_L],max_flow*.73, mape_phrase, fontsize = 8)
                KGE_hv = hv.Text(Eval_df.index[text_bbox_L],max_flow*.63, kge_phrase, fontsize = 8)
                textbox_hv = hv.Rectangles([(Eval_df.index[bbox_L], max_flow*.56, Eval_df.index[-1], max_flow*.99)]).opts(color = 'white')

                Mod_NWIS_Scatter = hv.Scatter((Eval_df[NWIS_site_lab], Eval_df[Mod_reach_lab]), Obs_Discharge_lab, Mod_Discharge_lab).opts(tools = ['hover'], color = 'blue', xrotation=45)
                Mod_NWIS_one2one = hv.Curve((flow_range, flow_range)).opts(color = 'red', line_dash='dashed')


                #display((NWIS_hydrograph * Mod_hydrograph).opts(width=600, legend_position='top_left', tools=['hover']) + (Mod_NWIS_Scatter*Mod_NWIS_one2one).opts(shared_axes = False))
                display((NWIS_hydrograph * Mod_hydrograph*textbox_hv*RMSE_hv*Error_hv*MAPE_hv*KGE_hv).opts(width=600, legend_position='top_left', tools=['hover']) + (Mod_NWIS_Scatter*Mod_NWIS_one2one).opts(shared_axes = False))

            else:
                print('No data for NWIS site: ', str(NWIS_site_lab), ' skipping.')




    def Map_Plot_Eval(self, freq, supply):
        self.freq = freq

        if self.freq == 'D':
            self.units = 'cfs'
        else:
            self.units = 'Acre-Feet'

        yaxis = 'Streamflow (' + self.units +')'

        #Adjust for different time intervals here
        #Daily
        if self.freq == 'D':
            self.NWIS_data_resampled = self.NWIS_data.copy()
            self.Mod_data_resampled = self.Mod_data.copy()


        #Monthly, Quarterly, Annual
        if self.freq !='D':
            #NWIS
            self.NWIS_data_resampled = self.NWIS_data.copy()*self.cfsday_AFday
            self.NWIS_data_resampled = self.NWIS_data_resampled.resample(self.freq).sum()
            #Modeled
            self.Mod_data_resampled = self.Mod_data.copy()*self.cfsday_AFday
            self.Mod_data_resampled = self.Mod_data_resampled.resample(self.freq).sum()
            
        if supply == True:
            #NWIS
            #Get Columns names
            columns = self.NWIS_data_resampled.columns

            #set up cumulative monthly values
            self.NWIS_data_resampled['Year'] = self.NWIS_data_resampled.index.year

            self.NWIS_CumSum = pd.DataFrame(columns=columns)

            for site in columns:
                self.NWIS_CumSum[site] = self.NWIS_data_resampled.groupby(['Year'])[site].cumsum()

            #Model
            #Get Columns names
            columns = self.Mod_data_resampled.columns

            #set up cumulative monthly values
            self.Mod_data_resampled['Year'] = self.Mod_data_resampled.index.year

            self.Mod_CumSum = pd.DataFrame(columns=columns)

            for site in columns:
                self.Mod_CumSum[site] = self.Mod_data_resampled.groupby(['Year'])[site].cumsum()
                
            #set the Mod and NWIS resampled data == to the CumSum Df's
            self.NWIS_data_resampled = self.NWIS_CumSum
            self.Mod_data_resampled =self.Mod_CumSum

        print('Plotting monitoring station locations')
        cols =  ['NWIS_site_id', 'NWIS_sitename', 'NHD_reachid', 'dec_lat_va', 'dec_long_va', 'geometry']

        self.df_map = self.HUC_NWIS[cols]
        self.df_map.reset_index(inplace = True, drop = True) 
        #Get Centroid of watershed
        centeroid = self.df_map.dissolve().centroid

        # Create a Map instance
        m = folium.Map(location=[centeroid.y[0], centeroid.x[0]], tiles = 'Stamen Terrain', zoom_start=8, 
                       control_scale=True)
        #add legend to map
        colormap = cm.StepColormap(colors = ['r', 'orange',  'y', 'g'], vmin = -1, vmax = 1, index = [-1,-0.4,0,0.3,1])
        colormap.caption = 'Model Performance (KGE)'
        m.add_child(colormap)

        ax = AxisProperties(
        labels=PropertySet(
            angle=ValueRef(value=300),
            align=ValueRef(value='right')
                )
            )

        for i in np.arange(0, len(self.df_map),1):
            #get site information
            site = self.df_map['NWIS_site_id'][i]
            USGSsite = 'USGS station id: ' + site
            site_name = self.df_map['NWIS_sitename'][i]

            reach = self.df_map['NHD_reachid'][i]
            Modreach = self.model +' reach id: ' + str(reach)
            
   
            #get modeled and observed information for each site
            df = pd.DataFrame(self.NWIS_data_resampled[site])
            df = df.rename(columns = {site: USGSsite})
            df[Modreach] = pd.DataFrame(self.Mod_data_resampled[reach])
            
            #remove na values or 0
            df_narem = pd.DataFrame()
            df_narem['obs'] = self.NWIS_data_resampled[site].astype('float64')
            df_narem['mod'] = self.Mod_data_resampled[reach].astype('float64')
            df_narem['error'] = df_narem['obs'] - df_narem['mod']
            df_narem['P_error'] = abs(df_narem['error']/df_narem['obs'])*100
            #drop inf values
            df_narem.replace([np.inf, -np.inf], np.nan, inplace = True)
            df_narem.dropna(inplace = True)

            obs = df_narem['obs']
            mod = df_narem['mod']

            #calculate scoring
            kge, r, alpha, beta = he.evaluator(he.kge, mod.astype('float32'), obs.astype('float32'))
            
            #set the color of marker by model performance

            if kge[0] > 0.30:
                color = 'green'

            elif kge[0] > 0.0:
                color = 'yellow'

            elif kge[0] > -0.40:
                color = 'orange'

            else:
                color = 'red'
            
            
            title_size = 14

            #create graph and convert to json
            graph = vincent.Line(df, height=300, width=500)
            graph.axis_titles(x='Datetime', y=yaxis)
            graph.legend(title= site_name)
            graph.colors(brew='Set1')
            #graph.axes[0].properties = ax
            graph.x_axis_properties(title_size=title_size, title_offset=35,
                          label_angle=300, label_align='right', color=None)
            graph.y_axis_properties(title_size=title_size, title_offset=-30,
                          label_angle=None, label_align='right', color=None)

            data = json.loads(graph.to_json())

            #Add marker with point to map, https://fontawesome.com/v4/cheatsheet/
            lat = self.df_map['dec_lat_va'][i]
            long = self.df_map['dec_long_va'][i]
            mk = features.Marker([lat, long], icon=folium.Icon(color=color, icon = 'fa-navicon', prefix = 'fa'))
            p = folium.Popup("Hello")
            v = features.Vega(data, width="100%", height="100%")

            mk.add_child(p)
            p.add_child(v)
            m.add_child(mk)


        display(m)
          
        
        
        
class Reach_Eval():
    def __init__(self, model , NWIS_list, startDT, endDT, cwd):
        self = self
        #self.df =df
        self.startDT = startDT
        self.endDT = endDT
        self.cwd = cwd
        self.cms_to_cfs = 35.314666212661
        self.model = model
        self.NWIS_list = NWIS_list
        self.cfsday_AFday = 1.983
        self.freqkeys = {
                        'D': 'Daily',
                        'M': 'Monthly',
                        'Q': 'Quarterly',
                        'A': 'Annual'
                        }


    def date_range_list(self):
        # Return list of datetime.date objects between start_date and end_date (inclusive).
        date_list = []
        curr_date = pd.to_datetime(self.startDT)
        while curr_date <= pd.to_datetime(self.endDT):
            date_list.append(curr_date)
            curr_date += timedelta(days=1)
        self.dates = date_list

   
    '''
    Get WBD HUC data
    '''
    def get_NHD_Model_info(self):
        try:
            print('Getting geospatial information for NHD reaches')
           
            #Load streamstats and covert to geodataframe
            self.Streamstats = pd.read_hdf(self.cwd+'/Data/StreamStats/StreamStats3.h5', 'streamstats')
            self.Streamstats.drop_duplicates(subset = 'NWIS_site_id', inplace = True)

            #Convert to geodataframe
            self.StreamStats = gpd.GeoDataFrame(self.Streamstats, geometry=gpd.points_from_xy(self.Streamstats.dec_long_va, self.Streamstats.dec_lat_va))

            #Get streamstats information for each USGS location
            self.sites = pd.DataFrame()
            for site in self.NWIS_list:
                s = self.Streamstats[self.Streamstats['NWIS_site_id'] ==  site]
                
                NHD_NWIS_df = utils.crosswalk(usgs_site_codes=site)
                
                if NHD_NWIS_df.shape[0] == 0:
                    NHD_segment = np.NaN
                    print('No NHD reach for USGS site: ', site)
                     #drop na nhd locations
                    #self.sites = self.sites[self.sites['NWIS_site_id'] != site]
                    
                
                else:
                    NHD_segment = NHD_NWIS_df.nwm_feature_id.values[0]
                    
                s['NHD_reachid'] = NHD_segment
                
                self.sites = self.sites.append(s)
            
            print('Dropping USGS sites with no NHD reach')
            self.sites = self.sites.dropna(subset = 'NHD_reachid')
            self.sites.NHD_reachid = self.sites.NHD_reachid.astype(int)

        except KeyError:
            print('No monitoring stations in this NWIS location')
            
        


    def prepare_comparison(self):

        #prepare the daterange
        self.date_range_list()

        self.comparison_reaches = list(self.sites.NHD_reachid)
        self.NWIS_sites = list(self.sites.NWIS_site_id)


        self.NWIS_data = pd.DataFrame(columns = self.NWIS_sites)
        self.Mod_data = pd.DataFrame(columns = self.comparison_reaches)

        self.sites.state_id = self.sites.state_id.str.lower()
        #create a key/dict of site/state id
        NWIS_state_key =  dict(zip(self.sites.NWIS_site_id, 
                                  self.sites.state_id))                           


        Mod_state_key =  dict(zip(self.sites.NHD_reachid, 
                              self.sites.state_id))

        #for NWM, add similar workflow to get non-NWM data
        print('Getting ', self.model, ' data')
        pbar = ProgressBar()
        for site in pbar(self.comparison_reaches):
          #  print('Getting data for NWM: ', site)
            state = Mod_state_key[site]
            filepath = self.cwd+'/Data/'+self.model+'/NHD_segments_'+state+'.h5'
            try:

                format = '%Y-%m-%d %H:%M:%S'
                Mod_flow = pd.read_hdf(filepath, key = str(site))
                Mod_flow['time'] ='12:00:00' 
                Mod_flow['Datetime'] = pd.to_datetime(Mod_flow['Datetime']+ ' ' + Mod_flow['time'], format = format)
                Mod_flow.set_index('Datetime', inplace = True)
                Mod_flow = Mod_flow.loc[self.startDT:self.endDT]

                flow = self.model + '_flow'


                self.Mod_data[site] = Mod_flow[flow]

            except:
                print(self.model,' site: ', site, ' not in database, skipping')

        #reset comparison reaches
        self.Mod_data.dropna(axis = 1, inplace = True)
        self.comparison_reaches = self.Mod_data.columns



        #Get NWIS data
        print('Getting NWIS data')
        pbar = ProgressBar()
        for site in pbar(self.NWIS_sites):
            try:
                state = NWIS_state_key[site]
                NWIS_meanflow =  pd.read_hdf(self.cwd+'/Data/NWIS/NWIS_sites_'+state+'.h5', key = str(site))
                format = '%Y-%m-%d %H:%M:%S'
                NWIS_meanflow.drop_duplicates(subset = 'Datetime', inplace = True)                
                NWIS_meanflow['time'] ='12:00:00' 
                NWIS_meanflow['Datetime'] = pd.to_datetime(NWIS_meanflow['Datetime']+ ' ' + NWIS_meanflow['time'], format = format)
                NWIS_meanflow.set_index('Datetime', inplace = True)
                NWIS_meanflow = NWIS_meanflow.loc[self.startDT:self.endDT]
                self.NWIS_data[site] = np.nan


                #Adjust for different time intervals here
                #Daily
                #if self.freq =='D':
                self.NWIS_data[site] = NWIS_meanflow['USGS_flow']

            except:
                    print('USGS site ', site, ' not in database, skipping')
                    #remove item from list
                    self.sites = self.sites[self.HUC_NWIS['NWIS_site_id'] !=  site]
        #reset NWIS sites   
        self.sites.reset_index(drop = True, inplace = True)
        self.NWIS_sites = self.sites['NWIS_site_id']

        #need to get the date range of NWIS and adjust modeled flow
        NWIS_dates = self.NWIS_data.index
        self.Mod_data = self.Mod_data.loc[NWIS_dates[0]:NWIS_dates[-1]]

        self.NWIS_column = self.NWIS_data.copy()
        self.NWIS_column = pd.DataFrame(self.NWIS_column.stack(), columns = ['NWIS_flow_cfs'])
        self.NWIS_column = self.NWIS_column.reset_index().drop('level_1',1)

        self.Mod_column = self.Mod_data.copy()
        col = self.model+'_flow_cfs'
        self.Mod_column = pd.DataFrame(self.Mod_column.stack(), columns = [col])
        self.Mod_column = self.Mod_column.reset_index().drop('level_1',1)



    def Interactive_Model_Eval(self, freq, supply):
        self.freq = freq

        if self.freq == 'D':
            self.units = 'cfs'
        else:
            self.units = 'Acre-Feet'

        #Adjust for different time intervals here
        #Daily
        if self.freq == 'D':
            self.NWIS_data_resampled = self.NWIS_data.copy()
            self.Mod_data_resampled = self.Mod_data.copy()

        #Monthly, Quarterly, Annual
        if self.freq !='D':
            #NWIS
            self.NWIS_data_resampled = self.NWIS_data.copy()*self.cfsday_AFday
            self.NWIS_data_resampled = self.NWIS_data_resampled.resample(self.freq).sum()
            #Modeled
            self.Mod_data_resampled = self.Mod_data.copy()*self.cfsday_AFday
            self.Mod_data_resampled = self.Mod_data_resampled.resample(self.freq).sum()
            
        if supply == True:
            #NWIS
            #Get Columns names
            columns = self.NWIS_data_resampled.columns

            #set up cumulative monthly values
            self.NWIS_data_resampled['Year'] = self.NWIS_data_resampled.index.year

            self.NWIS_CumSum = pd.DataFrame(columns=columns)

            for site in columns:
                self.NWIS_CumSum[site] = self.NWIS_data_resampled.groupby(['Year'])[site].cumsum()

            #Model
            #Get Columns names
            columns = self.Mod_data_resampled.columns

            #set up cumulative monthly values
            self.Mod_data_resampled['Year'] = self.Mod_data_resampled.index.year

            self.Mod_CumSum = pd.DataFrame(columns=columns)

            for site in columns:
                self.Mod_CumSum[site] = self.Mod_data_resampled.groupby(['Year'])[site].cumsum()
                
            #set the Mod and NWIS resampled data == to the CumSum Df's
            self.NWIS_data_resampled = self.NWIS_CumSum
            self.Mod_data_resampled =self.Mod_CumSum
            
            
        RMSE = []
        MAXERROR = []
        MAPE = []
        KGE = []

        for row in np.arange(0,len(self.sites),1):
            #Get NWIS id
            NWISid = self.sites['NWIS_site_id'][row]
            #Get Model reach id
            reachid = 'NHD_reachid'
            modid = self.sites[reachid][row]
            #get observed and prediction data
            obs = self.NWIS_data_resampled[NWISid]
            mod = self.Mod_data_resampled[modid]
            
            #remove na values or 0
            df = pd.DataFrame()
            df['obs'] = obs
            df['mod'] = mod.astype('float64')
            df['error'] = df['obs'] - df['mod']
            df['P_error'] = abs(df['error']/df['obs'])*100
            #drop inf values
            df.replace([np.inf, -np.inf], np.nan, inplace = True)
            df.dropna(inplace = True)
            
            obs = df['obs']
            mod = df['mod']

            #calculate scoring
            rmse = round(mean_squared_error(obs, mod, squared=False))
            maxerror = round(max_error(obs, mod))
            mape = df.P_error.mean()
            kge, r, alpha, beta = he.evaluator(he.kge, mod.astype('float32'), obs.astype('float32'))

            RMSE.append(rmse)
            MAXERROR.append(maxerror)
            MAPE.append(mape)
            KGE.append(kge[0])

        #Connect model evaluation to a DF, add in relevant information concerning LULC
        Eval = pd.DataFrame()
        Eval['NWIS_site_id'] = self.sites['NWIS_site_id']
        Eval[reachid] = self.sites[reachid]
        Eval['Location'] = self.sites['NWIS_sitename']
        Eval['RMSE'] = RMSE
        Eval['MaxError'] = MAXERROR
        Eval['MAPE'] = MAPE
        Eval['KGE'] = KGE
        Eval['Drainage_area_mi2'] = self.sites['Drainage_area_mi2']
        Eval['Mean_Basin_Elev_ft'] = self.sites['Mean_Basin_Elev_ft']
        Eval['Perc_Forest'] = self.sites['Perc_Forest']
        Eval['Perc_Imperv'] = self.sites['Perc_Imperv']
        Eval['Perc_Herbace'] = self.sites['Perc_Herbace']
        Eval['Mean_Ann_Precip_in'] = self.sites['Mean_Ann_Precip_in']
        Eval['Ann_low_cfs'] = self.sites['Ann_low_cfs']
        Eval['Ann_mean_cfs'] = self.sites['Ann_mean_cfs']
        Eval['Ann_hi_cfs'] = self.sites['Ann_hi_cfs']
        Eval['Location'] = self.sites.NWIS_sitename

        #sort dataframe and reindex
        self.Eval = Eval.sort_values('KGE', ascending = False).reset_index(drop = True)    
        #display evaluation DF
        display(self.Eval)
        
        #plot the model performance vs LULC to identify any relationships indicating where/why model performance
        #does well or poor
        #make all very negative KGE values -1
        self.Eval['KGE'][self.Eval['KGE'] < -1] = -1

        fig, ax = plt.subplots(3, 3, figsize = (11,11))
        fig.suptitle('Watershed Charcteristics vs. Model Performance', fontsize = 16)

        ax1 = ['Drainage_area_mi2', 'Mean_Basin_Elev_ft', 'Perc_Forest']
        for var in np.arange(0,len(ax1),1):
            variable = ax1[var]
            
            #remove na values for variables to make trendline
            cols = ['KGE', variable]
            df = self.Eval[cols]
            df.dropna(axis = 0, inplace = True)
            x = df['KGE']
            y = df[variable]
            ax[0,var].scatter(x = x, y = y)
            ax[0,var].set_ylabel(ax1[var])
            #add trendline
            #calculate equation for trendline
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                #add trendline to plot
                ax[0,var].plot(x, p(x), color = 'r', linestyle = '--')
            except:
                pass

        ax2 = ['Perc_Imperv', 'Perc_Herbace', 'Mean_Ann_Precip_in']
        for var in np.arange(0,len(ax2),1):
            variable = ax2[var]
             #remove na values for variables to make trendline
            cols = ['KGE', variable]
            df = self.Eval[cols]
            df.dropna(axis = 0, inplace = True)
            x = df['KGE']
            y = df[variable]
            ax[1,var].scatter(x = x, y = y)
            ax[1,var].set_ylabel(ax2[var])
            #add trendline
            try:
                #calculate equation for trendline
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                #add trendline to plot
                ax[1,var].plot(x, p(x), color = 'r', linestyle = '--')
            except:
                pass

        ax3 = ['Ann_low_cfs', 'Ann_mean_cfs', 'Ann_hi_cfs']
        for var in np.arange(0,len(ax3),1):
            variable = ax3[var]
             #remove na values for variables to make trendline
            cols = ['KGE', variable]
            df = self.Eval[cols]
            df.dropna(axis = 0, inplace = True)
            x = df['KGE']
            y = df[variable]
            ax[2,var].scatter(x = x, y = y)
            ax[2,var].set_xlabel('Model Performance (KGE)')
            ax[2,var].set_ylabel(ax3[var])
            #add trendline
            #calculate equation for trendline
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                #add trendline to plot
                ax[2,var].plot(x, p(x), color = 'r', linestyle = '--')
            except:
                pass


        plt.tight_layout()
        plt.show()
        
        num_figs = len(self.Eval)
        self.sites.reset_index(inplace = True, drop = True)
        for i in np.arange(0,num_figs,1):

            reach = self.Eval[reachid][i]
            site = self.Eval['NWIS_site_id'][i]
            #print(site, reach)
            sitename = self.Eval.Location[i]

            plot_title = self.freqkeys[self.freq]+ ' (' + self.units +') \n Performance of ' + self.model +' predictions, reach: ' + str(reach) + '\n USGS:' + str(site) +' ' + str(sitename)
            #plot_title = self.sites['name'][0] +' Basin' + ' ' + self.freqkeys[self.freq]+ ' (' + self.units +') \n Performance of ' + self.model +' predictions, reach: ' + str(reach) + '\n USGS:' + str(site) +' ' + str(sitename)


            NWIS_site_lab = 'USGS: ' + str(site)
            Mod_reach_lab = self.model + ': ' + str(reach)

            Eval_cols = [NWIS_site_lab, Mod_reach_lab]

            #Adjust for different time intervals here
            #Daily

            Eval_df = pd.DataFrame(index = self.NWIS_data_resampled.index, columns = Eval_cols)
            Eval_df[Mod_reach_lab] = self.Mod_data_resampled[reach]
            Eval_df[NWIS_site_lab] = self.NWIS_data_resampled[site]


            Eval_df = Eval_df.dropna()

            if Eval_df.shape[0] > 0:

                #need to have datetime fixed
                Eval_df = Eval_df.reset_index()
                Eval_df['Datetime'] = pd.to_datetime(Eval_df['Datetime'])
                Eval_df.set_index('Datetime', inplace = True, drop = True)
                
                #get observed and prediction data
                obs = Eval_df[NWIS_site_lab]
                mod = Eval_df[Mod_reach_lab]

                #remove na values or 0
                df = pd.DataFrame()
                df['obs'] = obs
                df['mod'] = mod.astype('float64')
                df['error'] = df['obs'] - df['mod']
                df['P_error'] = abs(df['error']/df['obs'])*100
                #drop inf values
                df.replace([np.inf, -np.inf], np.nan, inplace = True)
                df.dropna(inplace = True)

                obs = df['obs']
                mod = df['mod']

                #calculate scoring
                rmse = round(mean_squared_error(obs, mod, squared=False))
                maxerror = round(max_error(obs, mod))
                MAPE = round(mean_absolute_percentage_error(obs, mod)*100)
                kge, r, alpha, beta = he.evaluator(he.kge, mod.astype('float32'), obs.astype('float32'))
                
                #set limit to MAPE error
                if MAPE > 1000:
                    MAPE ='> 1000'

                rmse_phrase = 'RMSE: ' + str(rmse) +' ' +  self.units
                error_phrase = 'Max Error: ' + str(maxerror) +' ' + self.units
                mape_phrase = 'MAPE: ' + str(MAPE) + '%'
                kge_phrase = 'kge: ' + str(round(kge[0],2))
                
                max_flow = max(max(Eval_df[NWIS_site_lab]), max(Eval_df[Mod_reach_lab]))
                min_flow = min(min(Eval_df[NWIS_site_lab]), min(Eval_df[Mod_reach_lab]))

                flow_range = np.arange(min_flow, max_flow, (max_flow-min_flow)/100)

                if self.freq == 'A':
                    bbox_L = -int(round((len(Eval_df)*.32),0))
                    text_bbox_L = -int(round((len(Eval_df)*.18),0))
                    
                else:
                    bbox_L = -int(round((len(Eval_df)*.32),0))
                    text_bbox_L = -int(round((len(Eval_df)*.16),0))

                Discharge_lab = 'Discharge (' +self.units +')'
                Obs_Discharge_lab = ' Observed Discharge (' +self.units +')'
                Mod_Discharge_lab = self.model +' Discharge (' +self.units +')'


                NWIS_hydrograph = hv.Curve((Eval_df.index, Eval_df[NWIS_site_lab]), 'DateTime', Discharge_lab, label = NWIS_site_lab).opts(title = plot_title, tools = ['hover'], color = 'orange')
                Mod_hydrograph = hv.Curve((Eval_df.index, Eval_df[Mod_reach_lab]), 'DateTime', Discharge_lab, label = Mod_reach_lab).opts(tools = ['hover'], color = 'blue')
                RMSE_hv = hv.Text(Eval_df.index[text_bbox_L],max_flow*.93, rmse_phrase, fontsize = 8)
                Error_hv = hv.Text(Eval_df.index[text_bbox_L],max_flow*.83, error_phrase, fontsize = 8)
                MAPE_hv = hv.Text(Eval_df.index[text_bbox_L],max_flow*.73, mape_phrase, fontsize = 8)
                KGE_hv = hv.Text(Eval_df.index[text_bbox_L],max_flow*.63, kge_phrase, fontsize = 8)
                textbox_hv = hv.Rectangles([(Eval_df.index[bbox_L], max_flow*.56, Eval_df.index[-1], max_flow*.99)]).opts(color = 'white')

                Mod_NWIS_Scatter = hv.Scatter((Eval_df[NWIS_site_lab], Eval_df[Mod_reach_lab]), Obs_Discharge_lab, Mod_Discharge_lab).opts(tools = ['hover'], color = 'blue', xrotation=45)
                Mod_NWIS_one2one = hv.Curve((flow_range, flow_range)).opts(color = 'red', line_dash='dashed')


                #display((NWIS_hydrograph * Mod_hydrograph).opts(width=600, legend_position='top_left', tools=['hover']) + (Mod_NWIS_Scatter*Mod_NWIS_one2one).opts(shared_axes = False))
                display((NWIS_hydrograph * Mod_hydrograph*textbox_hv*RMSE_hv*Error_hv*MAPE_hv*KGE_hv).opts(width=600, legend_position='top_left', tools=['hover']) + (Mod_NWIS_Scatter*Mod_NWIS_one2one).opts(shared_axes = False))

            else:
                print('No data for NWIS site: ', str(NWIS_site_lab), ' skipping.')






    def Map_Plot_Eval(self, freq, supply):
        self.freq = freq

        if self.freq == 'D':
            self.units = 'cfs'
        else:
            self.units = 'Acre-Feet'

        yaxis = 'Streamflow (' + self.units +')'

        #Adjust for different time intervals here
        #Daily
        if self.freq == 'D':
            self.NWIS_data_resampled = self.NWIS_data.copy()
            self.Mod_data_resampled = self.Mod_data.copy()


        #Monthly, Quarterly, Annual
        if self.freq !='D':
            #NWIS
            self.NWIS_data_resampled = self.NWIS_data.copy()*self.cfsday_AFday
            self.NWIS_data_resampled = self.NWIS_data_resampled.resample(self.freq).sum()
            #Modeled
            self.Mod_data_resampled = self.Mod_data.copy()*self.cfsday_AFday
            self.Mod_data_resampled = self.Mod_data_resampled.resample(self.freq).sum()
            
        if supply == True:
            #NWIS
            #Get Columns names
            columns = self.NWIS_data_resampled.columns

            #set up cumulative monthly values
            self.NWIS_data_resampled['Year'] = self.NWIS_data_resampled.index.year

            self.NWIS_CumSum = pd.DataFrame(columns=columns)

            for site in columns:
                self.NWIS_CumSum[site] = self.NWIS_data_resampled.groupby(['Year'])[site].cumsum()

            #Model
            #Get Columns names
            columns = self.Mod_data_resampled.columns

            #set up cumulative monthly values
            self.Mod_data_resampled['Year'] = self.Mod_data_resampled.index.year

            self.Mod_CumSum = pd.DataFrame(columns=columns)

            for site in columns:
                self.Mod_CumSum[site] = self.Mod_data_resampled.groupby(['Year'])[site].cumsum()
                
            #set the Mod and NWIS resampled data == to the CumSum Df's
            self.NWIS_data_resampled = self.NWIS_CumSum
            self.Mod_data_resampled =self.Mod_CumSum
            

        print('Plotting monitoring station locations')
        cols =  ['NWIS_site_id', 'NWIS_sitename', 'NHD_reachid', 'dec_lat_va', 'dec_long_va', 'geometry']

        self.df_map = self.sites[cols]
        self.df_map.reset_index(inplace = True, drop = True) 
        #Get Centroid of watershed
        self.df_map = gpd.GeoDataFrame(self.df_map, geometry=gpd.points_from_xy(self.df_map.dec_long_va, self.df_map.dec_lat_va))

        centeroid = self.df_map.dissolve().centroid

        # Create a Map instance
        m = folium.Map(location=[centeroid.y[0], centeroid.x[0]], tiles = 'Stamen Terrain', zoom_start=8, 
                       control_scale=True)
        #add legend to map
        colormap = cm.StepColormap(colors = ['r', 'orange',  'y', 'g'], vmin = -1, vmax = 1, index = [-1,-0.4,0,0.3,1])
        colormap.caption = 'Model Performance (KGE)'
        m.add_child(colormap)

        ax = AxisProperties(
        labels=PropertySet(
            angle=ValueRef(value=300),
            align=ValueRef(value='right')
                )
            )

        for i in np.arange(0, len(self.df_map),1):
            #get site information
            site = self.df_map['NWIS_site_id'][i]
            USGSsite = 'USGS station id: ' + site
            site_name = self.df_map['NWIS_sitename'][i]

            reach = self.df_map['NHD_reachid'][i]
            Modreach = self.model +' reach id: ' + str(reach)
            
 
            
            #get modeled and observed information for each site
            df = pd.DataFrame(self.NWIS_data_resampled[site])
            df = df.rename(columns = {site: USGSsite})
            df[Modreach] = pd.DataFrame(self.Mod_data_resampled[reach])
            
            #remove na values or 0, this evaluates the model only on NWIS observations
            df_narem = pd.DataFrame()
            df_narem['obs'] = self.NWIS_data_resampled[site].astype('float64')
            df_narem['mod'] = self.Mod_data_resampled[reach].astype('float64')
            df_narem['error'] = df_narem['obs'] - df_narem['mod']
            df_narem['P_error'] = abs(df_narem['error']/df_narem['obs'])*100
            #drop inf values
            df_narem.replace([np.inf, -np.inf], np.nan, inplace = True)
            df_narem.dropna(inplace = True)

            obs = df_narem['obs']
            mod = df_narem['mod']

            #calculate scoring
            kge, r, alpha, beta = he.evaluator(he.kge, mod.astype('float32'), obs.astype('float32'))
            
            #set the color of marker by model performance

            if kge[0] > 0.30:
                color = 'green'

            elif kge[0] > 0.0:
                color = 'yellow'

            elif kge[0] > -0.40:
                color = 'orange'

            else:
                color = 'red'
            
            
            title_size = 14

            #create graph and convert to json
            graph = vincent.Line(df, height=300, width=500)
            graph.axis_titles(x='Datetime', y=yaxis)
            graph.legend(title= site_name)
            graph.colors(brew='Set1')
            #graph.axes[0].properties = ax
            graph.x_axis_properties(title_size=title_size, title_offset=35,
                          label_angle=300, label_align='right', color=None)
            graph.y_axis_properties(title_size=title_size, title_offset=-30,
                          label_angle=None, label_align='right', color=None)

            data = json.loads(graph.to_json())

            #Add marker with point to map, https://fontawesome.com/v4/cheatsheet/
            lat = self.df_map['dec_lat_va'][i]
            long = self.df_map['dec_long_va'][i]
            mk = features.Marker([lat, long], icon=folium.Icon(color=color, icon = 'fa-navicon', prefix = 'fa'))
            p = folium.Popup("Hello")
            v = features.Vega(data, width="100%", height="100%")

            mk.add_child(p)
            p.add_child(v)
            m.add_child(mk)


        display(m)