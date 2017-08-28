pytry .\trial\bcm_trial.py --weight_filename "data\neg_bcm" --save --bcm_rate -0.0001
pytry .\trial\bcm_trial.py --weight_filename "data\super_neg_bcm" --save --bcm_rate -0.00000001
pytry .\trial\bcm_trial.py --weight_filename "data\ultra_neg_bcm" --save --bcm_rate -0.0000000001
pytry .\trial\bcm_trial.py --weight_filename "data\pos_bcm" --save --bcm_rate 0.0001
pytry .\trial\learn_trial.py --weight_filename "data\basic" --save
