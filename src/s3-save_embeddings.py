from toolbox import ExportEmbeddingsForAllEpochs, CustomLogger

logger = CustomLogger("./custom_logs")

ExportEmbeddingsForAllEpochs("./models/FacebookAI/roberta-base/002", logger = logger).\
    routine()