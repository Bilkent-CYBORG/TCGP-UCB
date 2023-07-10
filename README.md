
# Contextual Combinatorial Multi-output GP Bandits with Group Constraints  
  
This repository is the official implementation of Contextual Combinatorial Multi-output GP Bandits with Group Constraints, published at TMLR. 

![Performance of our algorithm compared with the SOTA in semi-synthetic simulations.](https://dub07pap001files.storage.live.com/y4mc6y7pNAbjsWJFAd7EI5nqukAdZQ6j4dcZE6dlfSjhV6_347XkVxqq6uEFFm8YDTiAY88gHtSnjebCQCYFIaou8bNdOmhgQCfg8fHil7R-5YvoBVyHzlqQD54AGHqTNCNm0XqeA8sDr7xb3aeuWrGTS7l1oCouTbL3_jv0lQU_VPVk9Qk1Q7OOTAAafd5wHvcLkffM8c6QeWfhZ7aqe5G5ZcwCs6Ob8ZAW_AEsxyE_2g?encodeFailures=1&width=2902&height=956)
  
## Requirements  
  
To install requirements:  
  
```setup  
pip install -r requirements.txt  
```  
  
We use the [gpflow](https://github.com/GPflow/GPflow) library for all GP-related computations and gpflow uses tensorflow.  
  
## Running the simulations  
We ran a total of two simulations, one presented in the main paper and one presented in the supplemental. Moreover, none of the algorithms that we implement and test do offline-learning, thus there is no 'training' to be done. However, to be able to repeat the simulations and also improve speed, we first generate the arm contexts, rewards, and other setup-related information and save them as HDF5. We provide the links to the generated datasets that we used at the bottom of this README file. By default, when you run the script (main.py), it re-generates new datasets and runs the simulations on them.  
  
### Main paper simulations (privacy aware federated learning and caching-aware movie recommendation)  
To run the main paper simulations, provide the argument `main` to the main.py script.  
  
```  
python main.py main  
```  
and to run the main paper simulations using pre-generated datasets , which must be in the root directory, use  
```  
python main.py main --use_saved_dataset  
```  
### Supplemental simulations I (Different zeta)  
To run the first supplemental simulation, use the argument `supp_1`.