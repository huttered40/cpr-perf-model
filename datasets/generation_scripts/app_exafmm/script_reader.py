# Goal here is to transform the first 3 columns by dividing out by the process count

fp = open('kt1.csv','r')
fpw = open('kt1_mod.csv','w')
lines = fp.readlines()
count=0
for line in lines[0:]:
    x = line.split(',')
    fpw.write('%s,%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s'%(x[0],x[1],x[2],x[3],int(x[4])*int(x[2]),x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17]))
    count += 1
fp.close()
fpw.close()
