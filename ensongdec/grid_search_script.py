# Call main.py from CLI

# /home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_raw_HVC_z_w12m7_20.json
# /home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_raw_RA_z_w12m7_20.json
# /home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_trajectories_HVC_z_w12m7_20.json
# /home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_trajectories_RA_z_w12m7_20.json

# /home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_raw_HVC_z_r12r13_21.json
# /home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_raw_RA_z_r12r13_21.json
# /home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_trajectories_HVC_z_r12r13_21.json
# /home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_trajectories_RA_z_r12r13_21.json


# BIRD 1 - RAW, TRAJECTORIES
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_raw_HVC_z_w12m7_20.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_raw_RA_z_w12m7_20.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_trajectories_HVC_z_w12m7_20.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_trajectories_RA_z_w12m7_20.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"


# BIRD 2 - RAW, TRAJECTORIES
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_raw_HVC_z_r12r13_21.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_raw_RA_z_r12r13_21.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_trajectories_HVC_z_r12r13_21.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_trajectories_RA_z_r12r13_21.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"

# BIRD 1 - LATENT STABILITY ANALYSIS
python main_latent_stability.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_w12m7_20/config_raw_HVC_z_w12m7_20_latent_stability.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main_latent_stability.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_w12m7_20/config_raw_RA_z_w12m7_20_latent_stability.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main_latent_stability.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_w12m7_20/config_trajectories_HVC_z_w12m7_20_latent_stability.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main_latent_stability.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_w12m7_20/config_trajectories_RA_z_w12m7_20_latent_stability.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"


# BIRD 2 - LATENT STABILITY ANALYSIS
python main_latent_stability.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_raw_HVC_z_r12r13_21_latent_stability.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main_latent_stability.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_raw_RA_z_r12r13_21_latent_stability.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main_latent_stability.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_trajectories_HVC_z_r12r13_21_latent_stability.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main_latent_stability.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_trajectories_RA_z_r12r13_21_latent_stability.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"


# BIRD 2 - THRESHOLDS
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/config_thresholds_RA_z_r12r13_21.json" --override_dict "{\"hidden_layer_sizes\": [[8],[32],[64], [128], [256], [16, 16], [64, 64], [128, 128], [256, 256]]}"

# BIRD 2 - DECIMATE
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_raw_HVC_z_r12r13_21_25%clusters.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_raw_HVC_z_r12r13_21_50%clusters.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_raw_HVC_z_r12r13_21_75%clusters.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_raw_RA_z_r12r13_21_25%clusters.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_raw_RA_z_r12r13_21_50%clusters.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_raw_RA_z_r12r13_21_75%clusters.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_trajectories_HVC_z_r12r13_21_25%clusters.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_trajectories_HVC_z_r12r13_21_50%clusters.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_trajectories_HVC_z_r12r13_21_75%clusters.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_trajectories_RA_z_r12r13_21_25%clusters.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_trajectories_RA_z_r12r13_21_50%clusters.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"
python main.py --config_filepath "/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/configs/z_r12r13_21/config_trajectories_RA_z_r12r13_21_75%clusters.json" --override_dict "{\"hidden_layer_sizes\": [[64, 64]]}"


