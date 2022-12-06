# Streamflow_Evaluator
The streamflow evaluator compares modeled streamflow to in situ USGS monitoring sites, with interactive visualizations supporting an in-depth analysis.

## Application Overview
National-scale streamflow modeling remains a modern challenge, as changes in the underlying hydrology from land use and land cover (LULC) change, anthropogentic streamflow modification, and general process components (reach length, hydrogeophysical processes, precipitation, temperature, etc) greatly influence  hydrological modeling.
In a changing climate, there is a need to anticipate flood intensity, impacts of groundwater depletion on streamflow, western montain low-flow events, eastern rain-on-snow events, storm-induced flooding, and other severe environmental problems that challenge the management of water resources.
Given the National Water Model (NWM) bridges the gap between the spatially coarse USGS streamflow observations by providing a near-continuous 2.7 million reach predictions of streamflow, there lies the potential to improve upon the capabilities of the model by characterizing predictive performance across the heterogeneity of processes and land covers present at the national scale. 
The python-based Streamflow_Evaluator package provides a tool to evaluate national hydrogrphy dataset (nhd) based model outputs with colocated USGS/NWIS streamflow monitorng stations (parameter: 060). 
The package contains three key methods for evaluation: state-based LULC, HUC level analysis, and USGS station-based analysis.
Below is a description of each method and application.
While designed to us the NWM version 2.1 retrospective dataset, with minimal modification the tool should be able to take in other model formulations.
By using the Streamflow_Evaluator tool, researchers can identify locations where a model may benefit from further training/calibration/parameterization or a need for new model processes/features (e.g., integration of reservoir release operaitons) to ultimately create new post-processing methods and/or hydrological modeling formulations to improve streamflow prediction capabilities with respect to modeling needs (e.g., stormflow, supply, emergency management, flooding, etc).   

### Data Access
The Streamflow_Evaluator requires access to USGS/NWIS and colocated model output data for comparision.
While all data is publically available through the respective agencies, data download for analysis prevents timely model evaluation. 
The Alabama Water Institute at the University of Alabama hosts NWM v2.1 for all colocated USGS monitoring stations, and with the package capable of accessing the observed and predicted data, the package supports a fast and repeatable tool for evaluating modeled streamflow performance.
Data access protocols will be provided through Google Cloud Services.

## Dependencies (versions, environments)
Python: Version 3.8 or later

### Required packages
The included .yaml file should set up the correct Streamflow_Evaluator environment.
Below is a list of required packages to operate the tool:

| os           |    hydrotools   |      pandas  |
|:-----------: | :-------------: | :----------: | 
|  numpy       |  matplotlib     | sklearn      |
|  hydroeval   |  dataretrieval  | streamstats  |
|  geopandas   |  Ipython        | warnings     |
|  progressbar |  datetime       | folium       |
|  mapclassify |  time           | jenkspy      |
|  hvplot      |  holoviews      | bokeh        |
|  branca      |  vincent        | json         |
|  proplot     |  pygeohydro     | pygeoutils   |
|  switfter    |  geopy          |              |

## Streamflow Evaluation Options
Each streamflow evaluation method requires similar inputs, including a start date, end date, and model.
There are currently three different evaluation classes, each providing the user a unique method for evaluating streamflow modeling performance:
- State Land Use - Land Cover Evaluation
- Hydrologic Unit Code(s)
- USGS site id(s)

For all examples the prediction are from the NWM v2.1 retrospective. 

### State Land Use - Land Cover Evaluation
To determine how LULC affects the predictive performance of streamflow models, the Streamflow_Evaluator uses StreamStats to categorize the watershed upstream of each USGS monitoring site by watershed charateristics.
Please see the State Land Use - Land Cover Evaluation.md readme to use the tool.

![LULC_mapping](https://user-images.githubusercontent.com/33735397/205775870-5efab8e2-57ce-4ecb-b6c1-012909ece220.PNG)


_Running the Streamflow_Evaluator LULC_Eval class loads, processes, and visualizes model performance for the state, category, and size of interest_

![LULC_mapping_highlight](https://user-images.githubusercontent.com/33735397/205776459-355507b4-2036-4eca-8bb3-fc88debbebef.PNG)

_By clicking on a marker a popup of the modeled vs. observed performance at the inputted temporal frequency will appear_

![LULC_holoviews](https://user-images.githubusercontent.com/33735397/205777709-65a8e6d8-0d7a-42e5-81b3-819462cb6e6a.PNG)

_The Streamflow_Evaluator supports an interactive engagement with model results_



### Streamflow evaluation by Hydrologic Unit Code (HUC)
