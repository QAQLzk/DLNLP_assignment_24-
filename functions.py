import random
from collections import Counter
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, DatasetDict
import numpy as np
import sentencepiece
import seaborn as sns
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification,TrainerCallback
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#ignore warning
import warnings
warnings.filterwarnings("ignore")

#Record training process
global current_epoch
    

# Set random seeds
def set_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Load data
def load_data(dataset_path):
    df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
    return df

# Data visualization
def visualize_data(df):

    # plot distribution of score
    fig, ax = plt.subplots(figsize=(8, 6)) 
    fig.suptitle('Distribution of Score', size=16)
    plot = sns.countplot(x='score', data=df, palette='hls', ax=ax)
    for p in plot.patches:
        ax.annotate(format(p.get_height()),  
                    (p.get_x() + p.get_width() / 2., p.get_height()),  
                    ha = 'center',  
                    va = 'center',  
                    xytext = (0, 10), 
                    textcoords = 'offset points',
                    fontsize = 10)

    plt.savefig("results/Distribution of Score.png")
    print("Plot: Distribution of score has saved")


    # plot TOP10 anchors
    top = Counter([anc for anc in df['anchor']])
    top = dict(top.most_common(10))

    plt.figure(figsize=(10, 6))

    sns.barplot(x=list(top.keys()), y=list(top.values()), palette='hls')
    plt.xticks(rotation=90)
    plt.title("Top 10 First Phrases (Anchor)", fontsize=20)
    plt.savefig("results/Top 10 Anchor.png")
    print("Plot: Top 10 Anchor has saved")


    # plot Distribution of Section
    df['section'] = df['context'].astype(str).str[0]
    df['classes'] = df['context'].astype(str).str[1:]

    sections = {"A" : "A - Human Necessities", 
                "B" : "B - Operations and Transport",
                "C" : "C - Chemistry and Metallurgy",
                "D" : "D - Textiles",
                "E" : "E - Fixed Constructions",
                "F" : "F- Mechanical Engineering",
                "G" : "G - Physics",
                "H" : "H - Electricity",
                "Y" : "Y - Emerging Cross-Sectional Technologies"}

    plt.figure(figsize=(10, 8))

    sns.countplot(x='section', data=df, palette='rainbow', order = list(sections.keys())[:-1])
    plt.xticks([0, 1,2, 3, 4, 5, 6, 7], list(sections.values())[:-1], rotation='vertical')
    plt.title("Distribution of Section", fontsize=20)

    plt.savefig("results/Distribution of Section.png")
    print("Plot: Distribution of Section has saved")



    # plot Distribution of Score with respect to Sections
    plt.figure(figsize=(10, 6))

    sns.histplot(x='score', hue='section', data=df, bins=10, multiple="stack")
    plt.title("Distribution of Score with respect to Sections", fontsize=20)

    plt.savefig("results/Distribution of score and sections.png")
    print("Plot: Distribution of Score with respect to Sections has saved")


def data_preprocessing(df, model_nm): 
    # imerge the input
    df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor
    # Transfer Dataframe to dataset
    ds = Dataset.from_pandas(df)
    # Tokenize
    tokz = AutoTokenizer.from_pretrained(model_nm)

    def tok_func(x): return tokz(x["input"])
    tok_ds = ds.map(tok_func, batched=True)

    # change score name for input format.
    tok_ds = tok_ds.rename_columns({'score':'labels'})

    # Seperate train, valid, test dataset 
    train_test_split = tok_ds.train_test_split(test_size=0.2, seed=777)

    validation_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=777)

    train_ds = train_test_split["train"]
    valid_ds = validation_test_split["train"]
    test_ds = validation_test_split["test"]
    return train_ds, valid_ds, test_ds, tokz



def model_setup(train_ds, valid_ds, tokz, model_nm ):

    # Training Parameter
    bs = 16
    epochs = 5
    lr = 2e-5

    training_stats = {
        'train_loss': [],
        'eval_pearson': []
    }

    # Perason correlation calculation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.squeeze(predictions)
        pearson_corr = pearsonr(predictions, labels)[0]
        training_stats['eval_pearson'].append(pearson_corr)
        return {'eval_pearson': pearson_corr}

    # record loss in each epoch
    class LossRecorder(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            current_epoch = state.epoch

        def on_log(self, args, state, control, logs=None, **kwargs):
            if 'loss' in logs:
                training_stats['train_loss'].append(logs['loss'])
    

    args = TrainingArguments(
                            output_dir='results/',            
                            evaluation_strategy="epoch",       
                            save_strategy="epoch",             
                            learning_rate=lr,
                            warmup_ratio=0.1, 
                            lr_scheduler_type='cosine', 
                            per_device_train_batch_size=bs, 
                            per_device_eval_batch_size=bs*2,
                            num_train_epochs=epochs, 
                            weight_decay=0.01, 
                            fp16=False,
                            report_to='none')

    model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

    trainer = Trainer(model, 
                    args, 
                    train_dataset=train_ds,
                    eval_dataset=valid_ds,
                    tokenizer=tokz, 
                    compute_metrics=compute_metrics,
                    callbacks=[LossRecorder()])
    

    return  trainer, training_stats



def plot_train_process(training_stats):
    # plot Pearson Correlation in each epoch
    plt.figure(figsize=(8, 5))
    epoch=range(0,6)
    plt.plot(epoch, training_stats['eval_pearson'], label='Pearson Correlation', color='green', marker='o')
    plt.title('Pearson Correlation over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/Pearson Correlation in Each Epoch.png')
    print("plot: Pearson Correlation in Training Process has saved")

    loss_epochs = [0.27, 0.55, 0.82, 1.1, 1.37, 1.64, 1.92, 2.19, 2.47, 2.74, 3.02, 3.29, 3.56, 3.84, 4.11, 4.39, 4.66, 4.93]
    eval_epochs =[1,2,3,4,5]
    eval_losses = [0.0312,0.0260, 0.0263, 0.0235, 0.0231]
    plt.figure(figsize=(8, 6))
    plt.plot(loss_epochs, training_stats['train_loss'], label='Training Loss', color='b')
    plt.plot(eval_epochs, eval_losses,label='Training Loss', color='g' )
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/Training Loss in Each Epoch.png')
    print("plot: Loss in Training Process has saved")

    max_pearson = max(training_stats['eval_pearson'])

    return max_pearson



def pred_plot(trainer, test_ds):
    #predict
    preds = trainer.predict(test_ds).predictions.astype(float)
    preds = np.clip(preds, 0, 1)

    # Evaution
    true_labels = np.array(test_ds['labels'])

    mse = mean_squared_error(true_labels, preds)
    mae = mean_absolute_error(true_labels, preds)
    r2 = r2_score(true_labels, preds)
    pearson_corr_test, p_value = pearsonr(true_labels,preds)

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    print("Pearson Correlation Coefficient of Test set:", pearson_corr_test)
    print("P-value:", p_value)


    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(true_labels, preds, alpha=0.3)
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], 'k--') 
    plt.title('Actual vs Predicted')
    plt.savefig('results/Scatter Plot.png')
    print("plot: Scatter Plot of Actual vs Predicted has saved")

    # Residuals Plot
    residuals = true_labels - preds
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, residuals, alpha=0.3)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.hlines(y=0, xmin=min(preds), xmax=max(preds), colors='red', linestyles='--')
    plt.title('Residuals Plot')
    plt.savefig('results/Residuals Plot.png')
    print("plot: Residuals Plot has saved")


    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title('Histogram of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('results/Histogram.png')
    print("plot: Histogram of Prediction Errors has saved")

    return pearson_corr_test







