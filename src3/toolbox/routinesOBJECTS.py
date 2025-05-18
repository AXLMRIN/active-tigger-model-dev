from . import DATA, optimize_classifier, CustomLogger, cMapper
import gc 
import pandas as pd 
    
class routine:
    def __init__(self, folder_name : str, custom_mapping : cMapper,
                n_sample_range : list|range, epoch_range : list|range, 
                GA_parameters : dict, classifier):
        # Files parameters
        self.folder_name : str = folder_name
        
        # Loop parameters
        self.mapper : cMapper = custom_mapping
        self.n_sample_range : list|range = n_sample_range;   
        self.epoch_range : list|range = epoch_range;         
        
        self.current_n_sample : int = None
        self.current_epoch : int = None

        # pyGAD parameters
        self.GAp : dict = GA_parameters
        self.classifier = classifier
        
        # Saving variables
        self.save : list[dict] = []

    def optimisation_loop(self):
        d, evaluation_time, optimum, value, optimizer, string_log = (None,) * 6 
        try : 
            # Create the instances 
            d = DATA(self.folder_name,self.current_epoch, self.current_n_sample)
            optimizer = optimize_classifier(d, self.classifier, self.GAp, self.mapper)
            # run the optimisation process
            optimum, value, evaluation_time, generations_completed = optimizer.run()
            # Save
            self.save.append({
                "filename" : self.folder_name,
                "n_samples" : self.current_n_sample,
                "epoch" : self.current_epoch,
                "time" : evaluation_time,
                "f1_macro" : float(value),
                **{self.mapper.keys[idx] : self.mapper.functions[idx](value)
                    for idx, value in enumerate(optimum)}
            })
            #Logs
            string_log = (f"{'%.0f'%(self.current_n_sample):<10}|"
                f"{'%.0f'%(self.current_epoch):<10}|"
                f"{'%.2f'%(evaluation_time):<10}|"
                f"{'%.3f'%(float(value)):<10}|"
                f"{'%.0f'%(generations_completed):<10}|")
            for idx, value in enumerate(optimum):
                string_log += f"{'{}'.format(self.mapper.functions[idx](value)):<10}|"
            
            print(string_log)

        except Exception as e: 
            # Save
            self.save.append({
                "filename" : self.folder_name,
                "n_samples" : self.current_n_sample,
                "epoch" : self.current_epoch,
                "time" : None,
                "f1_macro" : None,
                **{key : None for key in self.mapper.keys}
            })
            # Logs
            string_log = (f"{'%.0f'%(self.current_n_sample):<10}|"
                f"{'%.0f'%(self.current_epoch):<10}|"
                f"{'FAILED':<10}|"
                f"{'FAILED':<10}|")
            for _ in range(self.GAp['num_genes']):
                string_log += f"{'FAILED':<10}|"
            string_log += f"\tError : {e}"
            
            print(string_log)
        finally : 
            # Clean
            del d, evaluation_time, optimum, value, optimizer, string_log
            gc.collect()
    
    def run_all(self):
        width = 11 * (5 + self.GAp['num_genes'])
        print("===  " * (1 + width // 5 ))
        print(self.folder_name,'\n')
        # Update the current values
        for n_sample in self.n_sample_range:
            self.current_n_sample = n_sample
            print('-' * width)
            for epoch in self.epoch_range:
                self.current_epoch = epoch
                ###
                self.optimisation_loop()
                ###
            print('-' * 11 * (5 + self.GAp['num_genes']))
            print()

    def save_to_csv(self, filename : str):
        try : 
            # The file exists
            df = pd.read_csv(filename)
            df = pd.concat((df,pd.DataFrame(self.save)))
        except:
            # The file does not exist
            df = pd.DataFrame(self.save)
        df.to_csv(filename, index = False)
        CustomLogger().\
            notify_when_done((
                f"Processed data : {self.folder_name}\n"
                f"Result saved : {filename}"
            ))
