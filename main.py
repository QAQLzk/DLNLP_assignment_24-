from functions import *

#ignore warning
import warnings
warnings.filterwarnings("ignore")

dataset_path = "Datasets/us-patent-phrase-to-phrase-matching/"
model_nm = 'microsoft/deberta-v3-small'
#======================================================================================================================
# Set seed
set_seed
# Load dataset
df = load_data(dataset_path)

# Data Visualization
visualize_data(df)

# Data preprocessing
train_ds, valid_ds, test_ds, tokz  = data_preprocessing(df, model_nm)
# ======================================================================================================================

trainer, training_stats = model_setup(train_ds, valid_ds, tokz, model_nm)              # Build and Set up the model and parameters
trainer.train()                                                        # Training the model
train_pearson = plot_train_process(training_stats)
test_pearson = pred_plot(trainer, test_ds)




# ======================================================================================================================
# ## Print out your results with following format:
print('The highest Pearson score in valid:{},and in test{};'.format(train_pearson, test_pearson))
