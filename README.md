# Anomaly-detection-in-feature-space-for-detecting-changes-in-phytoplankton-populations
This repository contains the code for reproducing results reported in the paper ["Anomaly detection in feature space for detecting changes in phytoplankton populations", Frontiers in Marine Science, 2024](https://www.frontiersin.org/articles/10.3389/fmars.2023.1283265/full). 

If you happen to refer to our work or use this code, please refer to it through the following citation:

```bibtex
@ARTICLE{10.3389/fmars.2023.1283265,  
AUTHOR={Ciranni, Massimiliano and Odone, Francesca and Pastore, Vito Paolo},   	 
TITLE={Anomaly detection in feature space for detecting changes in phytoplankton populations},      	
JOURNAL={Frontiers in Marine Science},      	
VOLUME={10},           	
YEAR={2024},      	  
URL={https://www.frontiersin.org/articles/10.3389/fmars.2023.1283265},       	
DOI={10.3389/fmars.2023.1283265},      	
ISSN={2296-7745},      
ABSTRACT={Plankton organisms are fundamental components of the earth’s ecosystem. Zooplankton feeds on phytoplankton and is predated by fish and other aquatic animals, being at the core of the aquatic food chain. On the other hand, Phytoplankton has a crucial role in climate regulation, has produced almost 50% of the total oxygen in the atmosphere and it’s responsible for fixing around a quarter of the total earth’s carbon dioxide. Importantly, plankton can be regarded as a good indicator of environmental perturbations, as it can react to even slight environmental changes with corresponding modifications in morphology and behavior. At a population level, the biodiversity and the concentration of individuals of specific species may shift dramatically due to environmental changes. Thus, in this paper, we propose an anomaly detection-based framework to recognize heavy morphological changes in phytoplankton at a population level, starting from images acquired in situ. Given that an initial annotated dataset is available, we propose to build a parallel architecture training one anomaly detection algorithm for each available class on top of deep features extracted by a pre-trained Vision Transformer, further reduced in dimensionality with PCA. We later define global anomalies, corresponding to samples rejected by all the trained detectors, proposing to empirically identify a threshold based on global anomaly count over time as an indicator that can be used by field experts and institutions to investigate potential environmental perturbations. We use two publicly available datasets (WHOI22 and WHOI40) of grayscale microscopic images of phytoplankton collected with the Imaging FlowCytobot acquisition system to test the proposed approach, obtaining high performances in detecting both in-class and out-of-class samples. Finally, we build a dataset of 15 classes acquired by the WHOI across four years, showing that the proposed approach’s ability to identify anomalies is preserved when tested on images of the same classes acquired across a timespan of years.}
}
