
import json

import h5py
import numpy as np
import pandas as pd

"""
Write sets of experiment files to disk or read from there.
"""

class MetadataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__cls": "ndarray",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tolist(),
            }
        if isinstance(obj, np.generic): # NumPy scalars
            return obj.item()
        return super().default(obj)
    
def write_campaign(campaign, filename="out.h5"):
    """ Write to HDF5 """
    # Write the main DataFrames
    with pd.HDFStore(filename, mode="w") as store:
        for handle, df in campaign.items():
            store.put(handle + "/dataframe", df.drop(columns=["ir", "ir-stddev"]), track_times=False)

    # Reopen to write the metadata and spectra
    with h5py.File(filename, "a") as f:
        for handle, df in campaign.items():
            group = f[handle]
            group.create_dataset("ir", data=np.array(df["ir"].to_list()))
            group.create_dataset("ir-stddev", data=np.array(df["ir-stddev"].to_list()))
        
            json_str = json.dumps(df.attrs, cls=MetadataEncoder)
            group.create_dataset("json-metadata", data=json_str, dtype=h5py.string_dtype("utf-8"))

def json_decoder(obj):
    if obj.get("__cls") == "ndarray":
        arr = np.array(obj["data"], dtype=obj["dtype"])
        return arr.reshape(obj["shape"])
    return obj

def read_campaign(filename):
    with pd.HDFStore(filename, mode="r") as store:
        restored = {key.split("/")[1]: store[key] for key in store.keys()}

    # Reopen to fetch the attrs
    with h5py.File(filename, "r") as f:
        for handle, df in restored.items():
            group = f[handle]

            metadata = json.loads(group["json-metadata"].asstr()[()], object_hook=json_decoder)

            df.attrs = metadata

            ir = group["ir"][:]
            ir_stddev = group["ir-stddev"][:]

            df["ir"] = pd.Series([row for row in ir])
            df["ir-stddev"] = pd.Series([row for row in ir_stddev])

    return restored
