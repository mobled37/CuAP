import csv
import os

# load the csv file and split csv file into 18 files
def split_csv():

    # split the csv file into 18 files
    with open('soundclip/vggsound.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i in range(18):
            with open('soundclip/vggsound' + str(i) + '.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                if i == 17:
                    for j in range(11106):
                        writer.writerow(next(reader))
                else:
                    for j in range(11080):
                        writer.writerow(next(reader))
                # write the last row

    count = 0
    for i in range(18):
        # return the number of row the vggsound0.csv file
        with open(f'soundclip/vggsound{i}.csv', 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                count += 1

    if count == 199466:
        print('split successfully')
    else:
        print('split failed')

def make_directories():
    # make directories


    for i in range(18):
        if not os.path.exists(f'vggsound{i}'):
            os.mkdir(f'vggsound{i}')

if __name__ == '__main__':
    # split_csv()
    make_directories()