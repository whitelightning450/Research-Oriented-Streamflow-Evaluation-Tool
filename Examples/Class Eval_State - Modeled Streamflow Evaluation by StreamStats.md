![LULC_Eval_top_image](https://user-images.githubusercontent.com/33735397/206321617-354fbbe1-3a61-4be2-8234-daf95fd4d926.PNG)


# Modeled Streamflow Evaluation by StreamStats
To determine how LULC affects the predictive performance of streamflow models, the Streamflow_Evaluator uses StreamStats to categorize the watershed upstream of each USGS monitoring site by watershed charateristics.
The LULC_Eval class allows the user to evaluate modeled streamflow with observed in situ NWIS monitoring sites.
Please enter a start date, end date, frequency, state of interest, and model to compare (NWM v2.1 is set up).
Select the below classifications to evaluate model performance.


|Watershed Characteristic                    | Code               |
|:----------------------------:              |:-----------------: |
| Drainage Area (mi<sup>2</sup>)             | Drainage_area_mi2  |
| Mean Basin Elevation (ft)                  | Mean_Basin_Elev_ft |
| Percentage Forest Area (%)                 | Perc_Forest        |
| Percentage Developed Area (%)              | Perc_Develop       |
| Percentage Impervious Area (%)             | Perc_Imperv        |
| Percentage Herbacious Area (%)             | Perc_Herbace       |
| Percentage of Slope Area > 30 degrees (%)  | Perc_Slop_30       |
| Precipitation (in)                         | Mean_Ann_Precip_in |
| Mean Annual Low Flow (cfs)                 | Ann_low_cfs        |
| Mean Annual Flow (cfs)                     | Ann_mean_cfs       |
| Mean Annual High Flow (cfs)                | Ann_hi_cfs         | 



Start the Streamflow_Evaluator by loading it into your interactive python environment (tested and developed with Jupyter notebooks).  
![initiate_Streamflow_Evaluator](https://user-images.githubusercontent.com/33735397/206513716-2182975f-53af-4860-b502-e2ce38a74ca2.PNG)

_The Streamflow_Evaluator requires the Streamflow_Evaluator.py python script._

Initiate the Streamflow_Evaluator by inputting a start date, end date, state, model, and classification (see above table).
Dependent on data availability, current NWIS is from 1980 - present where available and NWM v2.1 retrospective is from 1980-2020.

![initiateSE_LULC](https://user-images.githubusercontent.com/33735397/205773388-befae3c2-9c48-43ca-ba0c-9d847299dc80.PNG)

_Inititate the Streamflow_Evaluator by inputting a start date, end date, state, model, and classification.
For the purpose of the example, we are looking at the state of Washington (wa)_

![initiate_Streamflow_Evaluator_LULC_class](https://user-images.githubusercontent.com/33735397/206513777-e16968fe-e280-427d-ab3e-91921cdc8a38.PNG)

_Loading and running the LULC_Eval class within the Streamflow_Evaluator._

The function loads the repective state's StreamStats, locates necessary NWIS sites and colocated NHD reaches.
Using the Jenks classification algorithm, the function categorizes the specified watershed charateristic into five catgories.
Per the example:
Categorical breaks for  Drainage_area_mi2 in utah are:  [0.00139, 385.86, 1036.1, 2065.05, 3300.0, 5613.41]

The .Map_Plot_Eval() function is the preliminary evaluation tool, loading data, processing to the temporal frequency of interest, and incorporating interactive mapping capabilities to visulize model performance through marker color coding to the calculated King Gupta Efficiency Coefficient (KGE) and by clicking on marker, a popup of the modeled and observed the respective site.
The Map_Plot_Eval function requires the following inputs: temporal frequency, dataframe, dataframe name.

|Category     | Dataframe code      |
| :----------:|:-------------------:|
| Very Small  | State_Eval.df_vsmall|
| Small       | State_Eval.df_small |
| Medium      | State_Eval.df_medium|
| Large       | State_Eval.df_large |
| Very Large  | State_Eval.df_vlarge|

|Temporal Frequency | Code     |
|:-----------------:| :-------:|
|Daily              | 'D'      |
|Monthly            | 'M'      |
|Quarterly          | 'Q'      |
|Annual             | 'A'      |


![LULC_mapping](https://user-images.githubusercontent.com/33735397/205775870-5efab8e2-57ce-4ecb-b6c1-012909ece220.PNG)


_Running the Streamflow_Evaluator LULC_Eval class loads, processes, and visualizes model performance for the state, category, and size of interest._

![LULC_mapping_highlight](https://user-images.githubusercontent.com/33735397/205776459-355507b4-2036-4eca-8bb3-fc88debbebef.PNG)

_By clicking on a marker a popup of the modeled vs. observed performance at the inputted temporal frequency will appear.
The NWM v2.1 demonstrates high model performance for reach 24281986 at USGS monitoring location 12089500._

![LULC_poor_perf](https://user-images.githubusercontent.com/33735397/206320576-7c8fc91a-4c75-4bd1-9cc2-12dc0ab22f4e.PNG)

_By clicking on a marker a popup of the modeled vs. observed performance at the inputted temporal frequency will appear.
The NWM v2.1 demonstrates poor model performance for reach 24255125 at USGS monitoring location 1218100._

The Interactive_Model_Eval fuction takes in the temporal frequency of interest to support a more in-depth analysis, including error metrics.
The underlying holoviews plotting package supports interactive engagement with the plot, such as zooming in on events and hovering to get exact values.

![LULC_holoviews](https://user-images.githubusercontent.com/33735397/205777709-65a8e6d8-0d7a-42e5-81b3-819462cb6e6a.PNG)

_The Streamflow_Evaluator supports an interactive engagement with model results.
The NWM demonstrates good model performance for reach 23017906 at USGS monitoring location 12424000._


