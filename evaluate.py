# importing necessary libraries
from torch import nn
import numpy as np
import joblib
import torch

# defining the model
def clp_network(trial=False, n_layers=-1, units=[]):
    
    custom_network = not trial and len(units) == n_layers

    if not custom_network: n_layers = trial.suggest_int("n_layers", 2, 5)
    layers = []

    in_features = 6
    
    for i in range(n_layers):
        
        if custom_network: out_features = units[i]
        else: out_features = trial.suggest_int("n_units_l{}".format(i), 4, 10)
        
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU())

        in_features = out_features
    
        
    layers.append(nn.Linear(in_features, 2))
    layers.append(nn.LeakyReLU())
    
    return nn.Sequential(*layers)

# loading the model
print('Please enter which model you want to use:')
print('\t1. Manual Model\n\t2. Automatic Model\n\t\t DEFAULT = Automatic Model')
mode = int(input())
if mode==1: model_dict_path = 'Models\\customer_loyalty_prediction.pt'
else: model_dict_path = 'Models\\customer_loyalty_prediction_autp.pt'
model_dict = torch.load(model_dict_path)

# extracting network architecture data
architecture = model_dict['architecture']
n_layers = architecture['n_layers']
units = []
for i in range(n_layers):
    units.append(architecture["n_units_l{}".format(i)])

# setting up the model
model = clp_network(False, n_layers, units)
model.load_state_dict(model_dict['model_state_dict'])
model.eval()

# loading the scaler model
data_scaler = joblib.load('Models\\data_scaler.gz')

# getting input from user
print('Please enter the following values:')
print('Debt1:', end='\t')
debt1 = float(input())
print('Credit:', end='\t')
credit = float(input())
print('Age:', end='\t')
age = float(input())
print('Work Experience:', end='\t')
exp = float(input())
print('Education:', end='\t')
edu = float(input())
print('Time of Residency:', end='\t')
red = float(input())

# processing the data
data = np.array([debt1, credit, age, edu, exp, red])
data = data.reshape(1, -1)
data = torch.tensor(data_scaler.transform(data))

# feeding the data to model
with torch.no_grad():
    output = model(data.float())

# outputing the predicted value
output = output.argmax(dim=1).sum().item()
print(output)