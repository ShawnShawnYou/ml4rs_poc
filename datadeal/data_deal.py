import os
import time
import random
class Order:
    def __init__(self, s):
        self.raw_str = s
        string_list = s.split(",")
        self.start_time = time.strptime(string_list[1], '%Y-%m-%d %H:%M:%S')
        self.end_time = time.strptime(string_list[2], '%Y-%m-%d %H:%M:%S')
        self.max_wait = 120+random.randint(1, 10)
        self.raw_str += ',' + str(self.max_wait)
        self.pickup_loc = (float(string_list[5]), float(string_list[6]))
        try:
            self.dropoff_loc = (float(string_list[9]), float(string_list[10]))
        except:
            self.dropoff_loc = (0, 0)
            print(string_list)


    def __str__(self):
        return self.raw_str



def month_to_day(month):
    data_file = "E:\workspace\A-Experiment-code\FORM_Experiment_Code\dataset\ExperimentData/raw_data/yellow_tripdata_2013-%02d.csv" % month
    target_folder = "D:/ExperimentData/raw_data/%02d" % month #第一个"%"后面的内容为显示的格式说明,第二个"%"后面为显示的内容来源
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    # else:
    #     return
    days = [[] for _ in range(40)]
    with open(data_file, "r", encoding="utf8") as f:
        print(f.readline().strip())
        print(f.readline().strip())
        content = f.readline().strip()  #strip用于去除字符串首尾的字符，默认是空格、\n、\t
        while len(content) > 2:
            o = Order(content)
            days[o.start_time.tm_mday-1].append(o)
            content = f.readline().strip()
    for i, day in enumerate(days[:31]):
        day.sort(key=lambda x: time.mktime(x.start_time))
        with open(os.path.join(target_folder, "%d.csv" % (i+1)), "w", encoding="utf8") as f:
            for o in day:
                f.write(str(o) +"\n")



if __name__ == '__main__':
    #for i in range(11, 13):
        month_to_day(7)
