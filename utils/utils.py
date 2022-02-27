import json
import datetime

def label_map_to_dictionary_converter(path):
    """
    This function takes in a path(string) to label_map.pbtxt file and returns a dictionary 
    constructed from the file at given path.

    Args:
        path(string): string that represents the relative path of the label_map.pbtxt file.

    Returns:
        dictionary: a dictionary constructed from the items listed in the given path. 
            Keys are the id numbers of the items. Values are the corresponding 
            display_name of the items.

    Example:
        Sample file content in the .pbtxt file:
                item {
                    name: "/m/01g317"
                    id: 1
                    display_name: "person"
                }
                item {
                    name: "/m/0199g"
                    id: 2
                    display_name: "bicycle"
                }
        Sample returned dictionary:
            {
                1: "person",
                2: "bicycle"
            }
    """
    label_dict = dict() # dictionary with key:id(int), value:display_name(str)
    
    #Read in the file
    try:
        with open(path,'r') as f:
            line = f.readline()
            while line != '':
                if line.find("item {") != -1: #if a new item exists
                    # Parse the current item's field values
                    _ = f.readline() # we don't need the name value
                    id = int(f.readline().split(':')[-1].strip())
                    display_name = f.readline().split(':')[-1].strip().split('"')[-2].strip()
                    # Create a dictionary with the parsed values
                    label_dict[id] = display_name

                line = f.readline() # skips closing curly bracket('}') of item{ }
            return label_dict
    except:
        print(f"Couldn't parse the file given at {path}")



def get_config_json(config_path):
    """
    Returns the dictionary converted version of the json in the config_path(str).
    
    Args:
        config_path(str): Path to the json file.
    Returns:
        config_json(dict): dictionary converted version of the json found in the config_path.
    """
    with open(config_path,'r') as f:
        config_json = json.load(f)
        return config_json

def get_current_time():
    """
    Return the current time in the following string format: [%Y/%m/%d, %H:%M:%S] (e.g. 2021/08/21, 22:25:47)
    """
    now = datetime.datetime.now()
    
    current_time = now.strftime("%Y/%m/%d, %H:%M:%S")
    return current_time