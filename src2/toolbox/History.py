class History:
    def __init__(self):
        self.train_loss_per_epoch : dict[int:list[float]] = {}
        self.train_loss_global : list[float] = []

    def append_loss_train(self, epoch : int, loss_value) -> None:
        self.train_loss_global.append(loss_value)
        
        if epoch not in self.train_loss_per_epoch:
            self.train_loss_per_epoch[epoch] = []
        self.train_loss_per_epoch[epoch].append(loss_value)

    def __str__(self) -> str:
        return (
            "History object"
        )