
# Dataset and training commands for the project.
# -------------------------------------------------------------
generate_ba_shapes_dataset:
	PYTHONPATH=. python3 script/generate_dataset.py --config config/dataset/ba_shapes.yaml

generate_zinc_dataset:
	PYTHONPATH=. python3 script/generate_dataset.py --config config/dataset/zinc_no2.yaml

test_train_graphormer:
	PYTHONPATH=. python3 script/test_train.py --dataset_config config/dataset/ba_shapes.yaml --model_config config/model/graphformer.yaml


# -------------------------------------------------------------
# Training commands for different models and datasets.
# -------------------------------------------------------------

train_ba_graphdps:
	PYTHONPATH=. python3 script/run_training.py --dataset config/dataset/ba_shapes.yaml --model config/model/graphgps.yaml --train config/train/ba_shape_graphdps.yaml

train__ba_graphformer:
	PYTHONPATH=. python3 script/run_training.py --dataset config/dataset/ba_shapes.yaml --model config/model/graphformer.yaml --train config/train/ba_shape_graphformer.yaml

train_zinc_graphdps:
	PYTHONPATH=. python3 script/run_training.py --dataset config/dataset/zinc_no2.yaml --model config/model/graphgps.yaml --train config/train/zinc_graphdps.yaml

train__zinc_graphformer:
	PYTHONPATH=. python3 script/run_training.py --dataset config/dataset/zinc_no2.yaml --model config/model/graphformer.yaml --train config/train/zinc_graphformer.yaml

# -------------------------------------------------------------
# Commands for creating counterfactual datasets and preparing EAP analyses.
# -------------------------------------------------------------

create_counterfactual_zinc:
	PYTHONPATH=. python3 script/create_counterfactual_dataset.py --config config/dataset/zinc_no2.yaml

create_counterfactual_ba_shape:
	PYTHONPATH=. python3 script/create_counterfactual_dataset.py --config config/dataset/ba_shapes.yaml


# -------------------------------------------------------------
# EAP preparation commands for different models, datasets and approaches.
# -------------------------------------------------------------

prepare_eap_ba_shape_graphgps_mpnn:
	PYTHONPATH=. python3 script/run_eap.py --config config/eap/graphgps_ba_mpnn.yaml 

prepare_eap_ba_shape_graphgps_attention:
	PYTHONPATH=. python3 script/run_eap.py --config config/eap/graphgps_ba_mpnn_attention.yaml

prepare_eap_ba_shape_graphgps_classic:
	PYTHONPATH=. python3 script/run_eap.py --config config/eap/graphgps_ba_classic.yaml

optimize_graphgps_ba_classic:
	PYTHONPATH=. python3 script/optimize_threshold.py --config config/eap/graphgps_ba_classic.yaml --tolerance 0.01

optimize_graphgps_ba_mpnn:
	PYTHONPATH=. python3 script/optimize_threshold.py --config config/eap/graphgps_ba_mpnn.yaml --tolerance 0.01

optimize_graphgps_ba_attention:
	PYTHONPATH=. python3 script/optimize_threshold.py --config config/eap/graphgps_ba_mpnn_attention.yaml --tolerance 0.01

optimize_graphformer_ba_classic:
	PYTHONPATH=. python3 script/optimize_threshold.py --config config/eap/graphformer_ba_classic.yaml --tolerance 0.01

optimize_graphformer_ba_mpnn:
	PYTHONPATH=. python3 script/optimize_threshold.py --config config/eap/graphformer_ba_mpnn.yaml --tolerance 0.01

optimize_graphformer_ba_attention:
	PYTHONPATH=. python3 script/optimize_threshold.py --config config/eap/graphformer_ba_mpnn_attention.yaml --tolerance 0.01