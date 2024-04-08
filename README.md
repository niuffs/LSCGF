## Learning Sparse and Continuous Graph Structures for Multivariate Time Series Forecasting
------

#### Requirements
- python 3
- see `requirements.txt`

#### Data Preparation

###### H5 File
Download the traffic data files for Los Angeles (METR-LA) and Bay Area (PEMS-BAY) from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN). Put into the `data/{METR-LA,PEMS-BAY}` folder.

###### TXT File
Download Solar-Energy, Traffic, Electricity, Exchange-rate datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Put into the `data/{solar_AL,traffic,electricity,exchange_rate}` folder.

#### Split dataset

Run the following commands to generate train/validation/test dataset at `data/{METR-LA,PEMS-BAY,solar_AL,traffic,electricity,exchange_rate}/{train,val,test}.npz`.

```SHELL
python generate_data.py 
```

#### Train Commands

* METR-LA
```SHELL
# Use METR-LA dataset
python train.py --dataset_dir=data/METR-LA --input_dim=2
```
* PEMS-BAY
```SHELL
# Use PEMS-BAY dataset
python train.py --dataset_dir=data/PEMS-BAY --input_dim=2
```
* Solar-Energy
```SHELL
# Use Solar-Energy dataset
python train.py --dataset_dir=data/solar_AL --input_dim=1
```
* Traffic
```SHELL
# Use Traffic dataset
python train.py --dataset_dir=data/traffic --input_dim=1
```
* Electricity
```shell
# Use Electricity dataset
python train.py --dataset_dir=data/electricity --input_dim=1
```
* Exchange-rate
```SHELL
# Use Exchange-rate dataset
python train.py --dataset_dir=data/exchange_rate --input_dim=1
```

#### Citation

If you use our model ```LSCGF``` in research, please cite [this paper](https://proceedings.mlr.press/v189/chen23a.html):

```
@inproceedings{chen2023balanced,
  title={Balanced spatial-temporal graph structure learning for multivariate time series forecasting: a trade-off between efficiency and flexibility},
  author={Chen, Weijun and Wang, Yanze and Du, Chengshuo and Jia, Zhenglong and Liu, Feng and Chen, Ran},
  booktitle={Asian Conference on Machine Learning},
  pages={185--200},
  year={2023},
  organization={PMLR}
}
```