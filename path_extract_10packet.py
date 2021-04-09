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
            takenstate = re.findall(r"taken\s\d+|never\sexecuted", line)
            if len(takenstate) != 0:
                if 'never executed' in takenstate:
                    path[a][i] = 0.5
                    i = i + 1

                else:
                    takenstate_string = ",".join(takenstate)
                    counts_list = re.findall(r"\d+", takenstate_string)
                    counts_list_int = list(map(int, counts_list))
                    #print(counts_list_int)
                    if(counts_list_int[0] == 0):
                        path[a][i] = 1.5
                        i = i + 1

                    elif(counts_list_int[0] == 1):
                        path[a][i] = 2.5
                        i = i + 1

                    elif(counts_list_int[0] == 2):
                        path[a][i] = 3.5
                        i = i + 1
                    elif(counts_list_int[0] == 3):
                        path[a][i] = 4.5
                        i = i + 1
                    elif(counts_list_int[0] >= 4 and counts_list_int[0] <= 7):
                        path[a][i] = 5.5
                        i = i + 1
                    elif(counts_list_int[0] >= 8 and counts_list_int[0] <= 15):
                        path[a][i] = 6.5
                        i = i + 1
                    elif(counts_list_int[0] >= 16 and counts_list_int[0] <= 31):
                        path[a][i] = 7.5
                        i = i + 1
                    elif (counts_list_int[0] >= 32 and counts_list_int[0] <= 127):
                        path[a][i] = 8.5
                        i = i + 1
                    else:
                        path[a][i] = 9.5
                        i = i + 1

        #print(path)
    a = a+1
print(path)
np.savetxt('extract_path.csv', path, delimiter = ',')


