# Call main.py from CLI

# /home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_raw_HVC_z_w12m7_20.json
# /home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_raw_RA_z_w12m7_20.json
# /home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_trajectories_HVC_z_w12m7_20.json
# /home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_trajectories_RA_z_w12m7_20.json

# /home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_raw_HVC_z_r12r13_21.json
# /home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_raw_RA_z_r12r13_21.json
# /home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_trajectories_HVC_z_r12r13_21.json
# /home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_trajectories_RA_z_r12r13_21.json


python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_raw_HVC_z_w12m7_20.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"

python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_raw_RA_z_w12m7_20.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"

python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_trajectories_HVC_z_w12m7_20.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"

python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_trajectories_RA_z_w12m7_20.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"



python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_raw_HVC_z_r12r13_21.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"

python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_raw_RA_z_r12r13_21.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"

python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_trajectories_HVC_z_r12r13_21.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"

python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/decoding_paper_analysis/configs/config_trajectories_RA_z_r12r13_21.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"
