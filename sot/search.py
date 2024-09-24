import os 
def search(path):
  files=os.listdir(path)   #查找路径下的所有的文件夹及文件
  for filee in  files:
    #   if filee == 'GOT-10k_Train_000033':
    #     import ipdb; ipdb.set_trace()
    if filee.endswith('.py') or filee.endswith('.txt'):
       continue
    f=str(path+'/'+filee)    #使用绝对路径
    #   if os.path.isdir(f):  #判断是文件夹还是文件
    if not os.listdir(f):  #判断文件夹是否为空
          print(str(filee))
    #   else:
    #     print('f',f)  
if __name__ =='__main__':
#   path = raw_input('input_path:')  #raw_input 函数数从命令输入
  search(str('/mnt/data2/xzz/sot/data/got10k/test'))