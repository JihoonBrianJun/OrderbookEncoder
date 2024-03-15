# UpbitTrade
This project aims to apply deep learning (Transformer-based models, in particular) methodologies for the real-time trading on cryptocurrency market.

Model is trained on the public data provided by Binance, and actual trading is executed on the Upbit (largest crypto exchange in Korea), leveraging the pre-trained model.

## Preliminary
Conda environments used for this project can be replicated by the following command.
```
conda env create -f requirements.yaml
conda activate hoon
```

## Data
Raw data files can be either downloaded directly from Binance (https://www.binance.com/landing/data), or using the Binance API.

Data products used in this project is `**Book Ticker**`(tick-level updates of the best bid and best ask on an order book) and `**Trades**`(executed transactions updated at tick level).

This project used Jan/01/2024 ~ Jan/06/2024 data for training, and Jan/07/2024 data for validation.

Data files used for training in this project can be downloaded using the following command.
```
gdown {to-be-updated}
```

## Data Preprocessing Pipeline
To be updated.

## Train result
Model configurations are shown in the table below.
|Hidden dimension (Transformer)|# heads|# Enc/Dec layers (Each)|
|---|---|---|
|64|2|2|

Train hyperparameters is summarized in the following table.
|# epoch|Batch Size|Learning Rate|Gamma (for StepLR)|
|---|---|---|---|
|30|16|1e-4|1|

After training, metrics evaluated on validation data were as follows:
<img src="assets/Screenshot 2024-03-15 at 9.42.12 PM.png" width="651px" height="150px" title="TrainResult" alt="TrainResult"></img><br/>

## Metrics
1. Correct Rate
    * Correct if predicted direction and the actual direction coincides, otherwise incorrect.


2. Recall
    * ${Num \ of \ correct \ predictions \ for \ cases \ in \ denominator} \over {Num \ of \ price \ increase \ greater \ than \ threshold}$


3. Precision (Strong)
    * ${Num \ of \ price \ increase \ among \ cases \ in \ denominator} \over {Num \ of \ model \ prediction \ greater \ than \ threshold}$

