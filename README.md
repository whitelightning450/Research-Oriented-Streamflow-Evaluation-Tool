# Streamflow_Evaluator
The streamflow evaluator compares modeled streamflow to in situ USGS monitoring sites, with interactive visualizations supporting an in-depth analysis.

## Application Overview
Changes in the underlying hydrology from land use and land cover (LULC) change are contributing to increased flood intensity, groundwater depletion, reduced streamflow levels, and other severe environmental problems that challenge the management of water resources.
Given the National Water Model (NWM) bridges the gap between the spatially coarse USGS streamflow observations by providing a near-continuous 2.7 million reach predictions of streamflow, there lies the potential to improve upon the capabilities of the model by characterizing predictive performance across the heterogeneity of LULC present at the national scale. 
To determine how LULC affects the predictive performance of the NWM, researchers with NOAA’s Cooperative Institute for Research to Operations in Hydrology (CIROH) develop a python-based NWM-LULC evaluation pipeline that categorizes HUC watersheds by the level of urban, rangeland, cropland, and natural LULC types and determines the predictive performance with the comparison of co-located USGS streamflow instrumentation.
We are applying this workflow to 100 HUC-12 watersheds across the United States, using the NWM version 2.1 retrospective dataset for the period spanning from 2010 to 2020 to benchmark influences of different LULC on the predictive performance of the NWM.
By using this NWM-LULC evaluation tool, researchers with CIROH can identify LULC types where the application of new post-processing methods and/or hydrological modeling formulations could improve upon the national scale prediction of streamflow.   
