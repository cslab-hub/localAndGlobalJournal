# localAndGlobalJournal
Experiment code for LASA and GCR on univariate UCR datasets for the ??? paper. It is based on tensorflow and seml experiments to handle the different configurations.


For LASA see also: https://github.com/cslab-hub/LocalTSMHAInterpretability
For GCR see also: https://github.com/cslab-hub/GlobalTimeSeriesCoherenceMatrices

# Files
mixModel.yaml - seml experiment configuration to train and evaluate the models 
mixModelTrain.py - experiment runs for the different models
presults.yaml - seml experiments to process all experiment results into 1 file
resultProcessing.py - experiment code for the presults.yaml config
modules - different modules for the different model types + helper

# Dependencies:
python==3.7.3
tensorflow-gpu==2.4.1
seaborn==0.10.1
scipy==1.7.3
scikit-learn==0.23.2
pandas==1.3.5
matplotlib==3.5.1
ipykernel==6.9.1

tensorflow_addons==0.14.0 
tensorflow_probability==0.12.2 
pyts==0.11.0 
uea_ucr_datasets==0.1.2 
dill==0.3.5.1 
antropy==0.1.4 
tslearn==0.5.2
sktime==0.9.0 


we suggest the following installation:
conda create -n tsTransformer python==3.7.3 tensorflow-gpu==2.4.1 seaborn==0.10.1 scipy==1.7.3 scikit-learn==0.23.2 pandas==1.3.5 matplotlib==3.5.1 ipykernel==6.9.1

pip install seml===0.3.6 tensorflow_addons==0.14.0 tensorflow_probability==0.12.2 pyts==0.11.0 uea_ucr_datasets==0.1.2 dill==0.3.5.1 antropy==0.1.4 tslearn==0.5.2

Sktime is a bit annoying with the dependencies, but stil works anyway for our purpose, thus we do it separately:
pip install sktime==0.9.0 
