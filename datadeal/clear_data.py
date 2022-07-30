
class clear_data:
    def __init__(self,file_name,output_filename):
        res = []
        self.output_filename = output_filename
        self.filename = file_name

    def clear(self):
        with open(self.output_filename,'w') as f,open(self.filename,'r') as f1:
            for line in f1.readlines():
                if line.strip()=='':
                    continue
                if self.nodelete(line.strip().split(',')):
                    f.write(line)

    def nodelete(self,line):
        if line[4] == '0' or (line[5] == '0' and line[6] == '0' and line[9] == '0' and line[10] == '0'):
            return False
        if line[5] == line[9] and line[6] == line[10]:
            return False
        return True


if __name__ == '__main__':
    data_file = "E:/workspace/A-Experiment-code/FORM_Experiment_Code/dataset/ExperimentData/raw_data/07/7.csv"
    target_file = "E:/workspace/A-Experiment-code/FORM_Experiment_Code/dataset/ExperimentData/raw_data/07/clear7.csv"
    x = clear_data(data_file, target_file)
    x.clear()
