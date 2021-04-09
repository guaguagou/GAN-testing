import re
import numpy as np
import sys
n = int(sys.argv[1])
path = np.zeros((n,594), dtype=np.float)
#file_list = ['11.txt', '12.txt', '13.txt', '14.txt', '15.txt', '16.txt', '17.txt', '18.txt', '19.txt', '20.txt']
a = 0
while a < n:
    i = 0
    s = './temporar_path/{}.txt'.format(a)
    print(s)
    with open(s) as f:
        for line in f:
            rssi = re.findall(r"taken\s\d+|never\sexecuted", line)
            if len(rssi) != 0:
                if 'never executed' in rssi:
                    #print(rssi)
                    path[a][i] = 2.5
                    i = i + 1
                else:
                    if 'taken 0' in rssi:
                        #print(rssi)
                        path[a][i] = 0.5
                        i = i + 1
                    else:
                        path[a][i] = 1.5
                        i = i + 1

        #print(path)
    a = a+1
print(path)
np.savetxt('extract_path.csv', path, delimiter = ',')
