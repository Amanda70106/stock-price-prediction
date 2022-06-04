import pandas as pd
import os

filename = input('Input the csv file name: ')
input_directory = os.path.abspath("../csv") + '/'
datasets = pd.read_csv(input_directory + filename)
output_directory = os.path.abspath('./output/') 
if not os.path.isdir(output_directory):
  os.makedirs(output_directory)