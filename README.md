# MOSAICFlow

![MOSAICFlow](overview.png)

This is the codebase for MOSAICFlow (MultimOdal Spatial Alignment and Integration with Conditional Flow matching), a computational method to perform physical and morphological alignment on spatial data with multimple modalities. 

There are three main functions:
1. `src/MOSAICFlow/physical/physical_align`: Given a pair of spatial slices, potentially from different modalities, performs physical alignment to affine register one slice onto the other, and compute a probabilistic mapping.
2. `src/MOSAICFlow/morphological/train_morphological_model`: Given a pair of physically aligned slices and the probabilistic mapping, learns a neural ODE flow model between the two slices for morphological alignment.
3. `src/MOSAICFlow/morphological/flow_morphology`: Given the trained model from `train_morphological_model` and a set of point coordinates, numerically intergrate the neural ODE to compute morphologically aligned coordinates for the input points.


## Usage
### Physical Alignment
To perform physical alignment, use the `physical_align` function:
```python
from src.MOSAICFlow.physical import physical_align

slice1_aligned, slice2_aligned, affine_transformation, mapping = physical_align(slice1, slice2, max_iter=10, alpha=0.1)
```
### Morphological Alignment
To perform morphological alignment, use the `train_morphological_model` and `flow_morphology` function:
```python
from src.MOSAICFlow.morphological import train_morphological_model, flow_morphology

max_iter = 50000 # Number of iterations to train the neural network
model_save_name = 'PATH/TO/model.pt' # Path to save the trained model
hidden_dims = [64, 64, 64] # List of integers specifying the architecture of the hidden layers of the flow model
lr = 0.005 # Learning rate, default 0.005
train_morphological_model(slice1, slice2, mapping, max_iter=50000, model_save_name='PATH/TO/model.pt', hidden_dims=hidden_dims, lr=lr)
```
### Example
We have provided tutorial notebooks for both physical alignment and morphological alignment in the `tutorials` folder.

## Note
This is a very preliminary version of the codebase to get you started on using our tool. We are actively working on the project and will include more functionalities soon.

## Contact
If you encounter any problem running the software, please contact Xinhao Liu at xl5434@princeton.edu or Hongyu Zheng at hz7140@princeton.edu.
