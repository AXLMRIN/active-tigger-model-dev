from torch import nn, float32, Tensor
from torch import dtype as torch_dtype

class IdeologySentenceClassifier(nn.Module):
    def __init__(self, in_features : int, out_features : int, 
                 hidden_layers : int | None = None, 
                 hidden_layers_size : int | None = None,
                 device : str = "cpu", dtype : torch_dtype = float32) -> None:
        super().__init__()

        self.in_features : int = in_features
        self.out_features : int = out_features
        self.__hidden_layers : int|None = hidden_layers
        self.__hidden_layers_size : int|None = hidden_layers_size
        self.device : str = device
        self.dtype : torch_dtype = dtype

        self.__with_hidden_layers = isinstance(self.__hidden_layers, int) &\
            isinstance(self.__hidden_layers_size, int)

        if self.__with_hidden_layers : 
            self.FirstLayer = nn.Linear(in_features=self.in_features, 
                out_features= self.__hidden_layers_size, bias = True, 
                device = self.device, dtype = self.dtype)
            self.HiddenLayers = [
                nn.Linear(in_features = self.__hidden_layers_size, 
                    out_features= self.__hidden_layers_size, bias = True, 
                    device = self.device, dtype = self.dtype
                ) for _ in range(self.__hidden_layers)
            ]
            self.LastLayer = nn.Linear(in_features = self.__hidden_layers_size, 
                out_features= self.out_features, bias = True, 
                device = self.device, dtype = self.dtype)
            
            self.__n_parameters : int = 0 +\
                (self.in_features + 1) * self.__hidden_layers_size +\
                self.__hidden_layers * (self.__hidden_layers_size + 1) *\
                     self.__hidden_layers_size +\
                (self.__hidden_layers_size + 1) * self.out_features

        else : 
            self.FirstLayer = nn.Linear(in_features=self.in_features, 
                out_features= self.out_features, bias = True, 
                device = self.device, dtype = self.dtype)
            
            self.__n_parameters : int = (self.in_features + 1) * self.out_features

    def forward(self, input : Tensor) -> Tensor:
        if self.__with_hidden_layers : 
            y_hidden : Tensor = self.FirstLayer(input)
            for hidden_layer in self.HiddenLayers : 
                y_hidden = hidden_layer(y_hidden)
            return self.LastLayer(y_hidden)
        else : 
            return self.FirstLayer(input)
        
    def __str__(self) -> str : 
        return (
            f"IdeologySentenceClassifier(nn.Module) :\n"
            f"On {self.device}, type : {self.dtype}\n"
            f"\t- dimension of  input : {self.in_features}\n"
            f"\t- dimension of output : {self.out_features}\n"
            f"\t- with {self.__hidden_layers} hidden layers of dimension {self.__hidden_layers_size}\n"
            f"\n"
            f"Total number of parameters : {self.__n_parameters}"
        )