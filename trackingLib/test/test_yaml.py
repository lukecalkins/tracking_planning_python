import yaml

file_name = 'config/init_info_planner.yaml'
with open(file_name, 'r') as file:
    dict = yaml.load(file)

    with open(dict['targetConfig'], 'r') as targ_file:
        dict_targ = yaml.load(targ_file)



a=3