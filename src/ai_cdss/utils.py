import yaml

# ---------------------------------------------------------------------
# MultiKey Dictionary

class MultiKeyDict(object):
    """Dictionary-like class that supports multiple keys mapping to the same value."""
    def __init__(self, **kwargs):
        self._keys = {}
        self._data = {}
        for k, v in kwargs.items():
            self[k] = v

    def __getitem__(self, key):
        try:
            return self._data[key]
        except KeyError:
            return self._data[self._keys[key]]
    
    def get(self, key, default=None):
        """Alias for __getitem__ to allow mkd.get('key') syntax."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def __setitem__(self, key, val):
        try:
            self._data[self._keys[key]] = val
        except KeyError:
            if isinstance(key, tuple):
               if not key:
                  raise ValueError('Empty tuple cannot be used as a key')
               key, other_keys = key[0], key[1:]
            else:
               other_keys = []
            self._data[key] = val
            for k in other_keys:
                self._keys[k] = key

    def __repr__(self):
        return f"MultiKeyDict(data={self._data}, keys={self._keys})"
    
    __str__ = __repr__ 

    def items(self):
        """Return a view of the primary keys and their values."""
        return self._data.items()

    def add_keys(self, to_key, new_keys):
        if to_key not in self._data:
            to_key = self._keys[to_key]
        for key in new_keys:
            self._keys[key] = to_key

    @classmethod
    def from_dict(cls, dic):
        result = cls()
        for key, val in dic.items():
            result[key] = val
        return result
    
    # --- YAML Serialization Methods ---
    def to_yaml(self, filepath):
        """Save MultiKeyDict to a YAML file in the new format."""
        keys_dict = {}
        for alias, primary_key in self._keys.items():
            if primary_key in keys_dict:
                keys_dict[primary_key].append(alias)
            else:
                keys_dict[primary_key] = [alias]

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump({"data": self._data, "keys": keys_dict}, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filepath):
        """Load MultiKeyDict from a YAML file in the new format."""
        with open(filepath, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        instance = cls()
        instance._data = obj["data"]
        instance._keys = {}

        for primary_key, aliases in obj["keys"].items():
            for alias in aliases:
                instance._keys[alias] = primary_key
        
        return instance

