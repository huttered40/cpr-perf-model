# Goal here is to transform the first 3 columns by dividing out by the process count

coarsening_type_dict = {0:0,3:1,6:2,8:3,10:4,21:5,22:6}
relax_type_list = {0:0,3:1,4:2,6:3,8:4,13:5,14:6,16:7,17:8,18:9}
interp_type_list = {0:0,2:1,3:2,4:3,5:4,6:5,8:6,9:7,12:8,13:9,14:10,16:11,17:12,18:13}

fp = open('kt0.csv','r')
fpw = open('kt0_mod.csv','w')
lines = fp.readlines()
count=0
for line in lines[0:]:
    x = line.split(',')
    fpw.write('%d,%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s'%(count,int(x[0])/int(x[6]),int(x[1])/int(x[7]),int(x[2])/int(x[8]),coarsening_type_dict[int(x[3])],relax_type_list[int(x[4])],interp_type_list[int(x[5])],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13]))
    count += 1
fp.close()
fpw.close()
