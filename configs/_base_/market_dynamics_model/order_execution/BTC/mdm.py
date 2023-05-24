market_dynamics_model = dict(
    data_path="data/order_execution/BTC/data.csv",
filter_strength=1,
slope_interval=[-0.01,0.01],
dynamic_number=5,
max_length_expectation=150,
OE_BTC=True,
PM='',
process_datafile_path='',
market_dynamic_labeling_visualization_paths='',
key_indicator='adjcp',
timestamp='date',
tic='tic',
labeling_method='quantile',
min_length_limit=12,
merging_metric='DTW_distance',
merging_threshold=0.001,
merging_dynamic_constraint=1
)