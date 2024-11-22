library(ScottKnottESD)
library(readxl)
# 同样要记得换工作路径
finalpath= 'E:\\Desktop\\Github desktop备份集\\Different-Unsupervised-Algorithms-for-Software-Defect-Prediction\\CXX_CLUSTRTING\\Clustering4UEADP-main\\RQ1\\DrawPicData\\'

skresultpath='E:\\Desktop\\Github desktop备份集\\Different-Unsupervised-Algorithms-for-Software-Defect-Prediction\\CXX_CLUSTRTING\\Clustering4UEADP-main\\RQ1\\output\\'


file_names <- list.files(finalpath, pattern = "^[^~].*\\.xlsx$")
for (i in 1:length(file_names)) {
  path=paste(finalpath,sep = "",file_names[i])
  
  #csv<- read.csv(file=path, header=TRUE, sep=",")
  csv<- read_excel(path)
  csv<-csv[-1]
  sk <- sk_esd(csv)
  #可能是我数据不够，导致我跑这里的时候，如果出现一列数据全是一样的，会报错，我就寄在这了
  
  
  #plot(sk)
  
  resultpath=paste(skresultpath,sep = "",file_names[i])
  resultpath=paste(resultpath,sep = "",".txt")
  print(resultpath)
  
  write.table (sk[["groups"]], resultpath) 
}

