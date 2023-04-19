import os
import fnmatch
import yaml

def yaml_read_directory(directory_path, count=10, page=0, sort_order='desc'):
    files = os.listdir(directory_path)
    files = [f for f in files if f.endswith('.yaml')]
    files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    if sort_order == "desc":
        files.reverse()
    total_count = len(files)
    start = (page-1) * count
    end = start + count
    files_batch = files[start:end]
    data = []
    for file_name in files_batch:
        file_path = os.path.join(directory_path, file_name)
        with open(file_path) as file:
            data.append(yaml.safe_load(file))
    return data, total_count


def yaml_search(directory, search_number):
    """
    Searches for a YAML file in the specified directory that matches the pattern run_x_.yaml,
    where x is the search number.
    Returns the first matching YAML file as a dictionary.
    """
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, f"run_{search_number}.yaml"):
            with open(os.path.join(directory, filename), 'r') as file:
                return yaml.safe_load(file)
    return None