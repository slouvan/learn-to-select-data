import os


dirname = "/Users/slouvan/sandbox/emnlp2017_bilstm_cnn_crf/result_analyze/"
base_exp_dir_1 = "MTL_MIT_Restaurant_NB_SENT_"
base_exp_dir_2 = "_CONLL_NB_SENT_SAME"

for i in [10, 20, 40, 80, 100, 200, 400, 800]:
    path = os.path.join(dirname, base_exp_dir_1+str(i)+base_exp_dir_2)
    #print("Path = {}".format(path))
    with open(os.path.join(path, "performance.out"), "r") as f:
        lines = f.readlines()
        last_line = lines[len(lines) - 1]
        field = last_line.split("\t")
        print(field[len(field) - 1], end="")
