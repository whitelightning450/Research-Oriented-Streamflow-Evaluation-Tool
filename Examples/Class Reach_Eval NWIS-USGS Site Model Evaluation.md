
![REach_Eval_top](https://user-images.githubusercontent.com/33735397/206324095-dcc59508-bf4c-43a8-9a19-4fc5c573e205.PNG)

# NHD - USGS Streamflow Evaluation
The Reach_Eval class allows the user to evaluate modeled streamflow with selected NWIS monitoring sites of interest. 
The user can input multiple USGS sites (e.g.,['10163000', '10155500', '10155200', '10154200'])
Enter a start date, end date, and model to compare (NWM v2.1 is set up).
NWM retrospective data spans from 1980 - 2020, USGS/NWIS data is location dependent.


Start the Streamflow_Evaluator by loading it into your interactive python environment (tested and developed with Jupyter notebooks).  
![starting_SE](https://user-images.githubusercontent.com/33735397/205772795-ca0f9d6d-37df-46b4-9631-3d40713d2ebe.PNG)

_The Streamflow_Evaluator requires the Streamflow_Evaluator.py python script._

Initiate the Streamflow_Evaluator by inputting a start date, end date, state, model, and classification (see above table).
Dependent on data availability, current NWIS is from 1980 - present where available and NWM v2.1 retrospective is from 1980-2020.

![Reach_Eval_initiate](https://user-images.githubusercontent.com/33735397/206324389-65592266-4eb5-46a0-9f14-57b59c91556c.PNG)

_Inititate the Streamflow_Evaluator by inputting a start date, end date, state, model, and classification.
Example using HUC id's ['10163000', '10155500', '10155200', '10154200']._

The .Map_Plot_Eval() function is the preliminary evaluation function, processing to the temporal frequency of interest, and incorporating interactive mapping capabilities to visulize model performance through marker color coding to the calculated King Gupta Efficiency Coefficient (KGE) and by clicking on marker, a popup of the modeled and observed the respective site.

![Reach_Eval_map](https://user-images.githubusercontent.com/33735397/206324582-39281f56-9640-4c8a-be35-88d454cffb4d.PNG)

_The mapping product visualizes model performance for the colocated USGS locations of interest_

![Reach_Eval_poor](https://user-images.githubusercontent.com/33735397/206324796-5502b585-b60d-4b85-a39a-21d228b3487b.PNG)

_By clicking on a marker a popup of the modeled vs. observed performance at the inputted temporal frequency will appear.
As shown, the NWM v2.1 for reach 10376596 and does a poor job of modeling the Provo River at Provo, UT, USGS site: 1016300

![Reach_Eval_good](https://user-images.githubusercontent.com/33735397/206324880-560fcb99-36b5-4380-bf0d-10f0b63eaf3f.PNG)

_By clicking on a marker a popup of the modeled vs. observed performance at the inputted temporal frequency will appear.
As shown, the NWM v2.1 for reach 10373692 and does a good job of modeling the Provo River near Woodland, UT, USGS site: 10154200

The Interactive_Model_Eval() fuction takes in the temporal frequency of interest to support a more in-depth analysis, including error metrics.
The underlying holoviews plotting package supports interactive engagement with the plot, such as zooming in on events and hovering to get exact values.

![HUC_Provo_Perf](https://user-images.githubusercontent.com/33735397/206318782-34e21c9d-70ff-4cac-86ea-b8a21ca7375f.PNG)

_The Streamflow_Evaluator supports an interactive engagement with model results.
By using the python Holoviews package, the user can click, drag, and fully interact with the plot.
The plot provides general model performance metrics._


