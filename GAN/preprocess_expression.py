import sys,math
from PIL import Image  
   
 # 计算两个坐标的距离  
def Distance(p1,p2):  
      dx = p2[0]- p1[0]  
      dy = p2[1]- p1[1]  
      return math.sqrt(dx*dx+dy*dy)  
   
 # 根据参数，求仿射变换矩阵和变换后的图像。  
def ScaleRotateTranslate(image, angle, center =None, new_center =None, scale =None, resample=Image.BICUBIC):  
      if (scale is None)and (center is None):  
            return image.rotate(angle=angle, resample=resample)  
      nx,ny = x,y = center  
      sx=sy=1.0  
      if new_center:  
            (nx,ny) = new_center  
      if scale:  
            (sx,sy) = (scale, scale)  
      cosine = math.cos(angle)  
      sine = math.sin(angle)  
      a = cosine/sx  
      b = sine/sx  
      c = x-nx*a-ny*b  
      d =-sine/sy  
      e = cosine/sy  
      f = y-nx*d-ny*e  
      return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)  
 # 根据所给的人脸图像，眼睛坐标位置，偏移比例，输出的大小，来进行裁剪。  
def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):  
      # calculate offsets in original image 计算在原始图像上的偏移。  
      offset_h = math.floor(float(offset_pct[0])*dest_sz[0])  
      offset_v = math.floor(float(offset_pct[1])*dest_sz[1])  
      # get the direction  计算眼睛的方向。  
      eye_direction = (eye_right[0]- eye_left[0], eye_right[1]- eye_left[1])  
      # calc rotation angle in radians  计算旋转的方向弧度。  
      rotation =-math.atan2(float(eye_direction[1]),float(eye_direction[0]))  
      # distance between them  # 计算两眼之间的距离。  
      dist = Distance(eye_left, eye_right)  
      # calculate the reference eye-width    计算最后输出的图像两只眼睛之间的距离。  
      reference = dest_sz[0]-2.0*offset_h  
      # scale factor   # 计算尺度因子。  
      scale =float(dist)/float(reference)  
      # rotate original around the left eye  # 原图像绕着左眼的坐标旋转。  
      image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)  
      # crop the rotated image  # 剪切  
      crop_xy = (eye_left[0]- scale*offset_h, eye_left[1]- scale*offset_v)  # 起点  
      crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)   # 大小  
      image = image.crop((int(crop_xy[0]),int(crop_xy[1]),int(crop_xy[0]+crop_size[0]),int(crop_xy[1]+crop_size[1])))  
      # resize it 重置大小  
      image = image.resize(dest_sz, Image.ANTIALIAS)  
      return image  
    
if __name__ =="__main__":  
      base="/media/fangxuyang/"
      f=open("/media/fangxuyang/F/firefoxdownload/CK+database/dataaug/6photo.txt")
      ff=open("/media/fangxuyang/F/firefoxdownload/CK+database/dataaug/6photoeyes.txt")
      labelf=open("/media/fangxuyang/F/firefoxdownload/CK+database/dataaug/6photolabel.txt")
      data=ff.read()
      eyeposition=data.split('\n')

      label_data=labelf.read()
      mylabel=label_data.split('\n')

      num=0;
      mybase='0'
      print('start')
      for line in f:
            num=num+1
            if(num==1):
                  continue
            line=line.strip('\n')
            if line[line.rfind('.')-1]!='1':
                  continue
            if line[line.rfind('/')+2:line.rfind('/')+5]==mybase:
                  continue
            mybase=line[line.rfind('/')+2:line.rfind('/')+5]
            print(mybase)
            if num%6==1:
                  path=base+line
                  image =  Image.open(path).convert('L')
                  eyeposition1=eyeposition[2*num-3].split()
                  eyeposition2=eyeposition[2*num-2].split()
                  if(eyeposition1[0]<eyeposition2[0]):
                        leftx =int(eyeposition1[0])
                        lefty=int(eyeposition1[1])
                        rightx=int(eyeposition2[0])
                        righty=int(eyeposition2[1])
                  else:
                        rightx =int(eyeposition1[0])
                        righty=int(eyeposition1[1])
                        leftx=int(eyeposition2[0])
                        lefty= int(eyeposition2[1])

                  thelabel=mylabel[int((num-2)/6)]
                  savefile='neutral'
                  if int(thelabel)==1:
                        savefile='angry'
                  if int(thelabel)==2:
                        savefile='contemptuous'
                  if int(thelabel)==3:
                        savefile='disgusted'
                  if int(thelabel)==4:
                        savefile='fearful'
                  if int(thelabel)==5:
                        savefile='happy'
                  if int(thelabel)==6:
                        savefile='sad'
                  if int(thelabel)==7:
                        savefile='surprised'
                  print('save')
                  savepath="/media/fangxuyang/F/firefoxdownload/RaFD/train/"+savefile+"/"+line[line.rfind('/')+1:]
                  saveimage=CropFace(image, eye_left=(leftx,lefty), eye_right=(rightx,righty), offset_pct=(0.38,0.5), dest_sz=(256,256))
                  saveimage.save(savepath)  


            #break