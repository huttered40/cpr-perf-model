# Goal here is to transform the first 3 columns by dividing out by the process count

order_type_dict = {'DGZ':0,'DZG':1,'GDZ':2,'GZD':3,'ZDG':4,'ZGD':5}

fp = open('kt0.csv','r')
fpw = open('kt0_mod.csv','w')
lines = fp.readlines()
count=0
for line in lines[0:]:
    x = line.split(',')
    fpw.write('%s,%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s'%(x[0],x[1],x[2],x[3],order_type_dict[x[4]],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20],x[21],x[22],x[23],x[24],x[25]))
    count += 1
fp.close()
fpw.close()
