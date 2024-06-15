# Replication code for "Learning Fair Optimal Classification Trees"
### PAPER LINK HERE


# Code

The following describes the structure of our scripts:
```
Code
 ├── FlowOCT -- our proposed method. See readme.txt for details on implementation
 ├── FlowOCT_kamiran -- our proposed method, adapted for a fair comparison to DADT (see Appendix Section A)
 ├── FlowOCT_kamiran_warm -- same as above, but with warm start capability
 ├── Kamiran -- Matlab code to implement DADT Run driver.m
 ├── MIP_DT_DIDI -- Code to implement RegOCT
 ├── fairlearn.py -- Script to implement fairlearn

Data Preprocess Code: each folder contains the raw data, as well as R code to preprocess said data
 ├── adult
 ├── compas
 ├── german

 Results
 ├── viz.ipynb -- code to produce Figure 1 from raw results
```

# R Packages needed
The following R packages are required to run the .R scripts:
- caret
- stringr
- outliers
- dplyr
- editrules
- mlr


## sessionInfo()

# Python Packages needed
The following python packages are required to run the .py scripts:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- fairlearn (for fairlearn.py)
- gurobipy (for all MIO-based methods)