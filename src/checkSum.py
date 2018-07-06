import random

def char_checksum(data, byteorder='little'):
    length = len(data)
    checksum = 0
    for i in range(0, length):
        x = int.from_bytes(data[i:i+1], byteorder, signed=True)
        if x>0 and checksum >0:
            checksum += x
            if checksum > 0x7F: # 上溢出
                checksum = (checksum&0x7F) - 0x80 # 取补码就是对应的负数值
        elif x<0 and checksum <0:
            checksum += x
            if checksum < -0x80: # 下溢出
                checksum &= 0x7F
        else:
            checksum +=x # 正负相加，不会溢出
        #print(checksum)    
    return checksum     

def uchar_checksum(data, byteorder='little'):  
    length = len(data)
    checksum = 0
    for i in range(0, length):
        checksum += int.from_bytes(data[i:i+1], byteorder, signed=False)
        checksum &= 0xFF # 强制截断        
    return checksum