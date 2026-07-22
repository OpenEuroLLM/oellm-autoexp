In these sweeps we suggest the HP grid per model size and token budget. 

The versions refer to 
1. initial HP grid derived from the English scaling experiments. 
2. Adjusted HP grid following initial results and seeing the initial centers are not optimum.

Besides this, we differentiate between HPCs as it was discovered on MN5 that the num of workers matters a lot for throughput and depending on model size and GBSZ.



/training/mn5 contains the same training sweep leo (v1), but with num workers set per HP pair
/training/leo contains the same training sweep mn5 (v1), but adjusted to the data split inconsistencies


mention something about the low mbs on mn5 (and leo?) due to memory consumption


