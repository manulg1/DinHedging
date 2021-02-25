# Dinamic Hedging Insurance method

This package allows users to combine a probabilistic simulation model and a hedging methodology in order to design dynamic portfolio insurance for aquaculture companies.

To do that, the main script (Multiproc_DH.py) has been divided into three sections (Data Collection, Simulation Function, and Testing process):
- The first process is prepared to collect data (CSV files) on the selling prices and sea temperatures of the farm location.
- Then, the simulation function, which is supported by the script called "DinInsurance.py", carry out the required steps to apply the developed methodology
- Lastly, due to the computational requirements, the test process is prepared for multiprocessing (running independent parallel processes).
