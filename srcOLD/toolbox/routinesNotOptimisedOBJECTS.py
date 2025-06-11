from . import DATA, optimize_classifier, CustomLogger, cMapper
from sklearn.metrics import f1_score
from time import time
import gc 
import pandas as pd 
    
class routineNotOptmisied:
    def __init__(self, 
        folder_name : str, 
        n_sample_range : list|range, 
        epoch_range : list|range,  
        classifier, 
        classifier_parameters : dict, 
        logger : CustomLogger, 
        print_logs : bool = False):

        # Files parameters
        self.folder_name : str = folder_name
        self.logger : CustomLogger = logger
        self.print_logs : bool = print_logs

        # Loop parameters
        self.n_sample_range : list|range = n_sample_range;   
        self.epoch_range : list|range = epoch_range;         
        
        self.current_n_sample : int = None
        self.current_epoch : int = None

        # classifier parameters
        self.classifier = classifier
        self.classifier_parameters = classifier_parameters
        
        # Saving variables
        self.save : list[dict] = []

    def optimisation_loop(self):
        d, evaluation_time, value, optimizer, string_log = (None,) * 5
        try : 
            # Create the instances 
            d = DATA(self.folder_name,self.current_epoch, self.current_n_sample)
            clf = self.classifier(**self.classifier_parameters)

            # fit the classifier
            t1 = time()
            clf.fit(d.X_train, d.y_train)
            t2 = time()
            evaluation_time = t2 - t1 

            # Evaluate the score
            value = f1_score(y_true=d.y_test, y_pred=clf.predict(d.X_test), average='macro')

            # Save
            self.save.append({
                "filename" : self.folder_name,
                "n_samples" : self.current_n_sample,
                "epoch" : self.current_epoch,
                "time" : evaluation_time,
                "f1_macro" : float(value),
                **{key : self.classifier_parameters[key]
                    for key in self.classifier_parameters}
            })
            #Logs
            string_log = (f"{'%.0f'%(self.current_n_sample):<10}|"
                f"{'%.0f'%(self.current_epoch):<10}|"
                f"{'%.2f'%(evaluation_time):<10}|"
                f"{'%.3f'%(float(value)):<10}|")
            for key in self.classifier_parameters:
                string_log += f"{'{}'.format(self.classifier_parameters[key]):<10}|"
            
            self.logger.log(string_log+"\n", self.print_logs)

        except Exception as e: 
            # Save
            self.save.append({
                "filename" : self.folder_name,
                "n_samples" : self.current_n_sample,
                "epoch" : self.current_epoch,
                "time" : None,
                "f1_macro" : None,
                **{key : None for key in self.classifier_parameters}
            })
            # Logs
            string_log = (f"{'%.0f'%(self.current_n_sample):<10}|"
                f"{'%.0f'%(self.current_epoch):<10}|"
                f"{'FAILED':<10}|"
                f"{'FAILED':<10}|"
                f"{'FAILED':<10}|")
            for _ in self.classifier_parameters:
                string_log += f"{'FAILED':<10}|"
            string_log += f"\tError : {e}"
            
            self.logger.log(string_log+"\n", self.print_logs)
        finally : 
            # Clean
            del d, evaluation_time, value, optimizer, string_log
            gc.collect()
    
    def run_all(self):
        width = 11 * (5 + len(self.classifier_parameters))
        print("===  " * (1 + width // 5 ))
        print(self.folder_name,'\n')
        self.logger.log("===  " * (1 + width // 5 )+"\n", False)
        self.logger.log(f'{self.folder_name}\n', False)

        # Update the current values
        for n_sample in self.n_sample_range:
            self.current_n_sample = n_sample
            self.logger.log('-' * width+"\n", self.print_logs)
            for epoch in self.epoch_range:
                self.current_epoch = epoch
                ###
                self.optimisation_loop()
                ###
            self.logger.log('-' * width+"\n", self.print_logs)

    def save_to_csv(self, filename : str):
        try : 
            # The file exists
            df = pd.read_csv(filename)
            df = pd.concat((df,pd.DataFrame(self.save)))
        except:
            # The file does not exist
            df = pd.DataFrame(self.save)
        df.to_csv(filename, index = False)