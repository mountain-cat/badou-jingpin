import csv
import datetime

from nn_pipeline.config import Config


def record_data(begintime, endtime, acc, sample_count):
    with open(Config['summary_table_path'], 'a+', encoding='UTF-8') as f:
        csv_write = csv.writer(f)
        data_row = [begintime.strftime('%Y-%m-%d %H:%M:%S'),endtime.strftime('%Y-%m-%d %H:%M:%S')]
        data_row += [Config["model_type"], Config["hidden_size"], Config['epoch'], Config['batch_size'],
                     Config['pooling_style']]
        print((begintime-endtime).seconds)
        print(sample_count)
        print(Config['epoch'])
        data_row += [acc, (begintime-endtime).seconds*100/(sample_count*Config['epoch'])]
        csv_write.writerow(data_row)
