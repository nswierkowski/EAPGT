generate_ba_shapes_dataset:
	PYTHONPATH=. python3 script/generate_dataset.py --config config/dataset/ba_shapes.yaml

generate_zinc_dataset:
	PYTHONPATH=. python3 script/generate_dataset.py --config config/dataset/zinc_no2.yaml

test_train_graphormer:
	PYTHONPATH=. python3 script/test_train.py --dataset_config config/dataset/ba_shapes.yaml --model_config config/model/graphformer.yaml

train_graphgps:
	PYTHONPATH=. python3 script/run_training.py --dataset config/dataset/ba_shapes.yaml --model config/model/graphgps.yaml --train config/train/default.yaml

train_graphformer:
	PYTHONPATH=. python3 script/run_training.py --dataset config/dataset/ba_shapes.yaml --model config/model/graphformer.yaml --train config/train/default.yaml