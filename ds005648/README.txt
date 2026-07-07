# README

Experiment details for Mapping object space dimensions: new insights from temporal dynamics. The main folder contains the raw MEG data for all participants in standard bids format. See references. 

The “sourcedata” folder contains the trial behavioral data collected during the EEG Session. The data in this folder follows the following trial structure:

	•	sourcedata
	⁃	sub-[participant number]_task-targets_events.csv: contains all the events for each trial in the EEG session, detailing what was shown on the screen

	•	sub-[participant number]:contains BIDS formatted raw EEG data
	⁃	sub-[participant name]_task-targets_events_short.tsv: information about the channels used and sampling rate for all trials 
	⁃	sub-[participant name]_task-targets_eeg.bdf: EEG raw data 


