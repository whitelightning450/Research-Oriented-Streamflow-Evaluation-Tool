![HUC_Eval_top](https://user-images.githubusercontent.com/33735397/206322410-ea0e210c-c805-4599-ad6b-704c76ba42ce.PNG)


# Modeled Streamflow Evaluation by Hydrologic Unit Code (HUC)
The HUC_Eval class allows the user to evaluate modeled streamflow with observed in situ NWIS monitoring sites 
for a watershed(s) of interest. 
The user can input multiple watersheds (e.g., Great Salt lake: ['1601', '1602']).
Please enter a start date, end date, watersheds and model to compare (NWM v2.1 is set up).
NWM retrospective data spans from 1980 - 2020, USGS/NWIS data is location dependent.

Use The National Map Watershed Boundary Dataset (WBD) to identify the HUC size and unit code of interest
https://apps.nationalmap.gov/downloader/ to locate HUC of interest


Start the Streamflow_Evaluator by loading it into your interactive python environment (tested and developed with Jupyter notebooks).  
![initiate_Streamflow_Evaluator](https://user-images.githubusercontent.com/33735397/206514204-4d981e4c-bb22-4274-b0d4-3e46393bc376.PNG)

_The Streamflow_Evaluator requires the Streamflow_Evaluator.py python script._

Initiate the Streamflow_Evaluator by inputting a start date, end date, state, model, and classification (see above table).
Dependent on data availability, current NWIS is from 1980 - present where available and NWM v2.1 retrospective is from 1980-2020.

![initiate_Streamflow_Evaluator_HUC_class](https://user-images.githubusercontent.com/33735397/206514251-f9c5ce96-de11-479c-9e4b-0fb6c6b69ecd.PNG)

_Inititate the Streamflow_Evaluator by inputting a start date, end date, state, model, and classification.
Example using the Great Salt Lake watershed, HUC id's 1601 and 1602._

Loading and running the HUC_Eval class within the Streamflow_Evaluator.

![HUC_Eval get data](https://user-images.githubusercontent.com/33735397/206317420-c484ef33-fb43-4305-a0a1-0bdad0af3031.PNG)

_The .get_NWM_info() and .prepare_comparison() functions load the respective watershed StreamStats, locate NWIS sites, and colocated NHD reaches within the define HUC boundaries._


The .Map_Plot_Eval() function is the preliminary evaluation function, processing to the temporal frequency of interest, and incorporating interactive mapping capabilities to visulize model performance through marker color coding to the calculated King Gupta Efficiency Coefficient (KGE) and by clicking on marker, a popup of the modeled and observed the respective site.


![HUC_Eval_GSL_map](https://user-images.githubusercontent.com/33735397/206317743-671cf913-bb4b-4ae2-8f2d-ef0864a42dbe.PNG)

_The mapping product visualizes model performance for the watershed of interest (Great Salt lake, HUC 1601,1602 shown)_

![PoorPerHUC](https://user-images.githubusercontent.com/33735397/206318263-7ee3d2c3-ad21-43ae-8112-e3b4ff2c4bbd.PNG)

_By clicking on a marker a popup of the modeled vs. observed performance at the inputted temporal frequency will appear.
As shown, the NWM v2.1 for reach 10395905 and does a poor job of modeling the South Willow Creek near Granstville, UT, USGS site: 10172800_

![GoodPerfHUc](https://user-images.githubusercontent.com/33735397/206318488-4afb36b3-2d8f-4778-9d6b-9f3dff770bef.PNG)

_By clicking on a marker a popup of the modeled vs. observed performance at the inputted temporal frequency will appear.
As shown, the NWM v2.1 for reach 10373692 and does a good job of modeling the Provo River near Woodland, UT, USGS site: 10154200_

The Interactive_Model_Eval fuction takes in the temporal frequency of interest to support a more in-depth analysis, including error metrics.
The underlying holoviews plotting package supports interactive engagement with the plot, such as zooming in on events and hovering to get exact values.

![HUC_Provo_Perf](https://user-images.githubusercontent.com/33735397/206318782-34e21c9d-70ff-4cac-86ea-b8a21ca7375f.PNG)

_The Streamflow_Evaluator supports an interactive engagement with model results.
By using the python Holoviews package, the user can click, drag, and fully interact with the plot.
The plot provides general model performance metrics._


