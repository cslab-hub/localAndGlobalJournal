# localAndGlobalJournal
Experiment code for LASA and GCR on univariate UCR datasets for the paper: Extracting Interpretable Local and Global Representations from Attention on Time Series. It is based on tensorflow and seml experiments to handle the different configurations.


For LASA see also: https://github.com/cslab-hub/LocalTSMHAInterpretability
For GCR see also: https://github.com/cslab-hub/GlobalTimeSeriesCoherenceMatrices

# Description
These experiments analyse Transformer Attention on univariate datasets from the UCR UEA repository with LASA and GCR. Both methods use herby, a symbolic approximate approach.
LASA is a local abstraction technique to improve interpretation by reducing the complexity of the data. In these experiments, we analyse the performance and multiple XAI metrics on LASA to improve our understanding of the method and Transformer Attention itself. As subvariant we introduce LASA-S, which tries to find Shapelets in the abstracted data. 
GCR on the other hand is a global interpretation method which represents the data in a coherent multidimensional way, showing how each symbol effects each other symbol at a specific input. We analyse the performance and multiple XAI metrics to further improve our understanding of the GCR and Transformer Attention. As subvariants we introduce the threshold-based GCR-T and the penalty-based GCR-P. We analyse the GCRs ability to approximate the task as well as the model.
The datasets are limited by univariate datasets with the maximal input length of 500. Each run per datasets is limited to 1 day run-time.

# Files
mixModel.yaml - seml experiment configuration to train and evaluate the models <br>
mixModelTrain.py - experiment runs for the different models<br>
presults.yaml - seml experiments to process all experiment results into 1 file<br>
resultProcessing.py - experiment code for the presults.yaml config<br>
modules - different modules for the different model types + helper<br>

# Dependencies:
python==3.7.3<br>
tensorflow-gpu==2.4.1<br>
seaborn==0.10.1<br>
scipy==1.7.3<br>
scikit-learn==0.23.2<br>
pandas==1.3.5<br>
matplotlib==3.5.1<br>
ipykernel==6.9.1<br>
<br>
tensorflow_addons==0.14.0 <br>
tensorflow_probability==0.12.2 <br>
pyts==0.11.0 <br>
uea_ucr_datasets==0.1.2 <br>
dill==0.3.5.1 <br>
antropy==0.1.4 <br>
tslearn==0.5.2<br>
sktime==0.9.0 <br>
<br>

We suggest the following installation:<br>
1: conda create -n tsTransformer python==3.7.3 tensorflow-gpu==2.4.1 seaborn==0.10.1 scipy==1.7.3 scikit-learn==0.23.2 pandas==1.3.5 matplotlib==3.5.1 ipykernel==6.9.1<br>
<br>
2: pip install seml===0.3.6 tensorflow_addons==0.14.0 tensorflow_probability==0.12.2 pyts==0.11.0 uea_ucr_datasets==0.1.2 dill==0.3.5.1 antropy==0.1.4 tslearn==0.5.2<br>
<br>
Sktime is a bit annoying with the dependencies, but stil works anyway for our purpose, thus we do it separately:<br>
3: pip install sktime==0.9.0 

# Cite and publications

This code represents the used model for the following publication: TODO<br>
<br>
If you use, build upon this work or if it helped in any other way, please cite the linked publication.
