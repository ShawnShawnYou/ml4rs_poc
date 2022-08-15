# 数据配置
month = 7
day = 7
data_dir_path = "E:/workspace/A-Experiment-code/FORM_Experiment_Code/dataset/ExperimentData/"
data_path = data_dir_path + "raw_data//%02d/clear%d.csv" % (month, day)

# 算法路径
algorithm_path = r'E:\workspace\A-Experiment-code\FORM_Experiment_Code\FORM'


FEATURE_PATH = "E:/workspace/A-Experiment-code/e_ride_test/file/features.txt"
PREDICT_RESULT_PATH = "E:/workspace/A-Experiment-code/e_ride_test/file/results.txt"

# 超参数
fragment = 60
base_wait_time = 60
wait_time_noise = 10
w = 5   # 1, 2, 5, 10
window_size = 60

# 训练
MINIBATCH_SIZE = 64


# 算法
total_round = 2000
algorithm = 1
with_G = False


# 随机种子
seed = 71437

# 司机数量
driverCount = 4000

#
takeTime = 60 * 3

# 区域分割
regionx = 10
regiony = 10


XREGION = (-74.01, -73.93)
YREGION = (40.70, 40.92)


NUM_GRID_X = 500
NUM_GRID_Y = 500



