import yaml

def yaml_to_fit(input_string):
    print yaml.dump(yaml.load(input_string))

