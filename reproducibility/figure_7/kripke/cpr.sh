python cpr.py --test_set_size 9000 --test_set_split_percentage 0.0 --interp_map 0,0,0,0,0,0,0,0,0 --response_transform 1 --training_file '/work2/05608/tg849075/app_ed/datasets/stampede2/kripke/kt0_nnodes1.csv' --test_file '/work2/05608/tg849075/app_ed/datasets/stampede2/kripke/kt0_nnodes1_test.csv' --output_file 'test_kripke_full_node.csv' --input_columns 1,2,3,4,5,9,10,14,15 --data_columns 24 --cell_spacing 1,1,0,0,1,0,0,0,0 --ngrid_pts 7,7,6,8,6,16,8,5,2 --cp_rank 32 --reg 1e-5 --build_extrapolation_model 0 --training_set_size 16384
