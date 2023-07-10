import os
from shutil import  copy
import  xlsxwriter
path = ('/data/LHX/bone/all/')
file = os.listdir(path)
n = 0
for i in file:
    old_name = path + file[n]
    # print(old_name)
    if file[n].split('-')[0] == 'Giant':
        num = int(file[n].split('-')[1])

        fold = (file[n].split('-')[2])
        fold = int(fold.split('.')[0])
        new_name = str(num) + '-' + str(fold) + '.jpg'
        if not os.path.exists('/data/LHX/bone/image-sp/' + str(num)):
            os.mkdir('/data/LHX/bone/image-sp/' + str(num))
        new_path = '/data/LHX/bone/image-sp/' + str(num)
        copy(old_name, new_path+'/'+new_name)
    if file[n].split('-')[0] == 'IMG':
        num = int(file[n].split('-')[1]) + 1000
        fold = (file[n].split('-')[2])
        fold = int(fold.split('.')[0])
        new_name = str(num) + '-' + str(fold) + '.jpg'
        if not os.path.exists('/data/LHX/bone/image-sp/' + str(num)):
            os.mkdir('/data/LHX/bone/image-sp/' + str(num))
        new_path = '/data/LHX/bone/image-sp/' + str(num)
        copy(old_name, new_path+'/'+new_name)
    n += 1

workbook = xlsxwriter.Workbook('/data/LHX/bone//label_all.xlsx')
worksheet = workbook.add_worksheet()
path = ('/data/LHX/bone/image-sp')
file = os.listdir(path)
n = 0
for i in file:
    print(n)
    if int(i) < 1000:
        worksheet.write(n, 0, int(i))
        worksheet.write(n, 1, 0)
    else:
        worksheet.write(n, 0, int(i))
        worksheet.write(n, 1, 1)
    n += 1
workbook.close()