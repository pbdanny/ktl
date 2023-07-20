from pathlib import Path as _Path_, _windows_flavour, _posix_flavour
import os
import json

class DBPath(_Path_):
    
    _flavour = _windows_flavour if os.name == 'nt' else _posix_flavour
    
    def __init__(self, input_file):
        if "dbfs:" in input_file:
            raise ValueError("DBPath accept only file API path style (path start with /dbfs/)")
        super().__init__()
        return
        
    def __repr__(self):
        return f"DBPath class : {self.as_posix()}"
    
    def file_api(self):
        rm_first_5_str = str(self.as_posix())[5:]
        return str("/dbfs"+rm_first_5_str)
    
    def spark_api(self):
        rm_first_5_str = str(self.as_posix())[5:]
        return str("dbfs:"+rm_first_5_str)


def conf_reader(path):
    f = open(path, "r")
    mapper = json.loads(f.read())
    f.close()
    return mapper

def save(df, path, format="delta", mode="overwrite", overwriteSchema=None):
    if overwriteSchema:
        df.write.format(format).mode(mode).option("overwriteSchema", True).save(
            path)
    else:
        df.write.format(format).mode(mode).save(path)
    print(">>> Saved to {}".format(path))