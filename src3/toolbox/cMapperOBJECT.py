class cMapper:
    def __init__(self,keys, functions):
        self.keys = keys
        self.functions = functions

    def __call__(self, idx, value):
        try: return {self.keys[idx] : self.functions[idx](value)}
        except: raise(KeyError, "custom mapping idx not right")