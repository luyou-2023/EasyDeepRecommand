import csv

def add_headers_to_file(input_filename, output_filename):
    # 定义表头
    headers = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]

    # 读取input_filename文件
    with open(input_filename, 'r') as infile:
        reader = csv.reader(infile, delimiter='\t')
        data = list(reader)

    # 写入output_filename文件，添加表头
    with open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)  # 写入表头
        writer.writerows(data)    # 写入数据

# 使用函数
data_path = '/Users/ctb/WorkSpace/EasyDeepRecommend/Dataset/criteo/train.txt'
output_path = '/Users/ctb/WorkSpace/EasyDeepRecommend/Dataset/criteo/train.csv'
add_headers_to_file(data_path, output_path)