
# TODO Réaliser un bandeau
# IMPORTS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# Third parties
from datasets import DatasetDict, Dataset

# FUNCTIONS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def storage_options():
    return {
        'client_kwargs': {'endpoint_url': 'https://minio-simple.lab.groupe-genes.fr'},
        'key': os.environ["AWS_ACCESS_KEY_ID"],
        'secret': os.environ["AWS_SECRET_ACCESS_KEY"],
        'token': os.environ["AWS_SESSION_TOKEN"]
    }


def split_test_train_valid(dataset : Dataset, proportion_train : float = 0.7,
    proportion_test : float = None, proportion_valid : float = None,
    shuffle : bool = True, seed : int = 42, print_proportions : bool = False
    ) -> DatasetDict:
    if (proportion_test is None) & (proportion_valid is None):
        proportion_test = (1 - proportion_train) / 2
        proportion_valid = proportion_test
    elif (proportion_test is None) : 
        proportion_test = 1 - proportion_train - proportion_valid
    elif (proportion_valid is None) :
        proportion_valid = 1 - proportion_train - proportion_test
    try:
        assert(proportion_train + proportion_test + proportion_valid == 1)
    except:
        print((
            "WARNING Wrong entry. Your entries :\n"
            "\t - proportion_train : {proportion_train}\n"
            "\t - proportion_test : {proportion_test}\n"
            "\t - proportion_valid : {proportion_valid}\n"
            "\n"
            "By default we use the tuple (0.7,0.15,0.15)"
        ))
        proportion_train, proportion_test, proportion_valid = 0.7,0.15,0.15

    ds_temp = dataset.train_test_split(test_size = proportion_test, 
                shuffle=shuffle, seed = seed)
    ds_temp2 = ds_temp["train"].train_test_split(
                        train_size = proportion_train / (1- proportion_test),
                        shuffle = shuffle, seed = seed
    )
    ds = DatasetDict({
        "train" : ds_temp2["train"],
        "validation" : ds_temp2["test"],
        "test" : ds_temp["test"],
    })
    
    if print_proportions : print_datasetdict_proportions(ds)
    return ds

def print_datasetdict_proportions(ds : DatasetDict):
    def proportion(name):
        return int(100 * len(ds[name]) / sum(ds.num_rows.values()))
    print("Répartition des datasets : ")
    print(f'| {"Dataset":^15}|{"Taille":^10}|{"Proportion":<7} (%)|')
    print("-" * 43)
    print(f'| {"Train":<15}|{len(ds["train"]):^10}|{proportion("train"):^14}|')
    print(f'| {"Validation":<15}|{len(ds["validation"]):^10}|{proportion("validation"):^14}|')
    print(f'| {"Test":<15}|{len(ds["test"]):^10}|{proportion("test"):^14}| ')