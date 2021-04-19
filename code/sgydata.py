import struct
import numpy as np
import random

def norm_sgy(data):
    maxval=max(max(data))
    minval=min(min(data))
    return [[(float(i)-minval)/float(maxval-minval) for i in j] for j in data]
def norm_sgy1(data):
    #maxval=max(max(data))
    #minval=min(min(data))
#    return [[(float(i)-min(j))/float(max(j)-min(j)) for i in j] for j in data]
    return [[float(i)/float(max(np.abs(j))) for i in j] for j in data]

def norm_sgy2(data):
    maxval=np.max(np.abs(data),axis=1)
    index=[i for i in range(len(data))]
    
    return [data[j]/maxval[j] for j in index]

def norm_tgy(data):
#    maxval=np.max(np.abs(data),axis=1)
    maxval=np.max(1000)
    index=[i for i in range(len(data))]
    return [data[j]/maxval for j in index]

def norm_ydata(data):
    maxval=np.max(np.abs(data),axis=0)
    minval=np.min(np.abs(data),axis=0)
    
#    maxval=np.max(data)
#    minval=np.min(data)
#    return [(j-minval[np.where(j)])/(maxval[np.where(j)]-minval[np.where(j)]) for j in data], maxval, minval
    return [j/maxval for j in data], maxval, minval
#    return [[(float(i)-float(minval))/float(maxval-minval) for i in j] for j in data]
    
def read_sgy(sgynam):
#    print "sgynam: "+sgynam
    try:
        binsgy = open(sgynam,'rb')
    except IOError:
        return 0,0,[]
    fhead=binsgy.read(3600);
#    print fhead[3213:3215]
    nr=struct.unpack(">H",fhead[3212:3214])
    print (nr)
    nsmp=struct.unpack(">H",fhead[3220:3222])
    print(nsmp)
    data = []
    for ir in range(0,nr[0]):
      trchead=binsgy.read(240)
      trcdata=binsgy.read(nsmp[0]*4)
      data1 = []
      for i in range(0,nsmp[0]):
       # print(trcdata[i*4:i*4+4])
        data1=data1+list(struct.unpack(">f",trcdata[i*4:i*4+4]))
      data.append(data1)
    print("read 1sgy end")
    binsgy.close()
    return nr,nsmp,data;

def read_egy(egynam):
    try:
        binegy = open(egynam,'rb')
    except IOError:
        return 0,[]
    data=[]
    fbinegy=binegy.read()
    ndata=len(fbinegy)//4
    for i in range(ndata):
        data1=[]
        data1=struct.unpack("f",fbinegy[i*4:i*4+4])
        data.append(data1)

    print("read 1egy end")
    binegy.close()
    return ndata, data

def read_tgy(tgynam):
    try:
        binegy = open(tgynam,'rb')
    except IOError:
        return 0,[]
    data=[]
    fbinegy=binegy.read()
    ndata=len(fbinegy)//4
    for i in range(ndata):
        data1=[]
        data1=struct.unpack("f",fbinegy[i*4:i*4+4])
        data.append(data1)

    print("read 1tgy end")
    binegy.close()
    return data

def load_data(sgynam='sgy',sgyf1=0,sgyt1=300,sgyf2=0,sgyt2=300,shuffle='true'):
    data= []
    ydata=[]
    for i in range(sgyf1,sgyt1):
       print(sgynam+"/event/event%04d.sgy" %(i))
       nr,nsmp,data1 = read_sgy(sgynam+"/event/event%04d.sgy" %(i));
       if nr != 0:
         data1=norm_sgy1(data1)
         data.append(data1)
         ydata.append(1);
       else:
         print('1 event sgy not found')
    for i in range(sgyf2,sgyt2):
       nr,nsmp,data1 = read_sgy(sgynam+"/noise/noise%04d.sgy" %(i));
       if nr != 0:
         data1=norm_sgy1(data1)
         data.append(data1)
         ydata.append(0);
       else:
         print('1 noise sgy not found')
    index=[i for i in range(len(ydata))]
    random.seed(7)
    if shuffle == 'true':
       random.shuffle(index)
    data = [data[i] for i in index]
    ydata = [ydata[i] for i in index]
    data=np.array(data)
    ydata=np.array(ydata)
    return data.shape[0],(data.shape[1]*data.shape[2]),data,ydata

def load_sgylist(sgylist,floc,shuffle='false'):
    data= []
    ydata=[]
    lines=open(sgylist,'r').readlines()
    lines2=open(floc,'r').readlines()
    for i in range(0,len(lines)):
       egynam=lines[i][:lines[i].find(' ')]
       print(egynam)
       ndata, data1 = read_egy(egynam)
       
       tgynam=lines2[i][:lines2[i].find(' ')]
       print(tgynam)
       data2 = read_tgy(tgynam);
       if ndata != 0:
         data2=norm_tgy(data2)
         data.append(data1)
         ydata.append(data2)
       else:
         print('1 event tgy not found')
    index=[i for i in range(len(ydata))]
    random.seed(7)
    if shuffle == 'true':
       random.shuffle(index)
    data = [data[i] for i in index]
    ydata = [ydata[i] for i in index]
    data=np.array(data)
    ydata=np.array(ydata)
    
    return [lines[i][:lines[i].find(' ')] for i in index], data,ydata

if __name__ == '__main__':
#     nr,nsmp,data=read_sgy()
#     print nr,nsmp,data[1:nsmp[0]]
      numsgy,len1,data,ydata=load_data()
      print (numsgy,len1,len(data),data[1],data.shape,ydata.shape)


