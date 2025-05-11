import csv

def append_list_to_csv(file_path, string_list):
    # 打开文件，以追加模式写入
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 将列表作为一行写入文件
        writer.writerow(string_list)

# 示例
file_path = 'output.csv'
string_list = ['字符串1', '字符串2', '字符串3']
append_list_to_csv(file_path, string_list)
string_list = ['字符串1', '字符串2', '字符串3']
append_list_to_csv(file_path, string_list)