class Config:
    def __init__(self):
        self.dataset_filename : str = "data/316_ideological_book_corpus/ibc.csv" # UPGRADE make it editable 
        self.dataset_label_col : str = "leaning"
        self.dataset_text_col : str = "sentence"
        self.seed : int = 2306406
        self.split_parameters : dict = {
            "proportion_train" : 0.7,
            "proportion_test" : 0.15, 
            "proportion_valid" : 0.15,
            "shuffle" : True, 
            "seed" : self.seed
        }
    def __str__(self):
        return (
            "Config object"
        )