import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    font = {'family': 'Times New Roman',
            'style': 'normal',
            'weight': 'normal',
            'color': 'black',
            'size': 12
            }
    plt.rc('font', family='Times New Roman')
    plt.figure()

    file_path = '/home/server/Lihu/results/curves/idea1/big_block_nums_bigdata/'
    file_list = ['1', '2', '4']
    for file_name in file_list:
        result = pd.read_csv(file_path+file_name+'.csv')
        data = result['psnr']
        plt.plot(data, label=file_name)

    plt.xlabel('Epoch', fontdict=font)
    plt.ylabel('PSNR', fontdict=font)
    plt.legend(loc='lower right')
    plt.show()