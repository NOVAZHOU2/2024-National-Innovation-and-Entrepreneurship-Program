library(ScottKnottESD)
library(readxl)
finalpath = "../Result/"
skresultpath = "../output/"
# finalpath= "C:/Users/Andre/Desktop/CUDP-master/discussion/Result_/"
# skresultpath='C:/Users/Andre/Desktop/CUDP-master/discussion/output_/'
file_names<- list.files(finalpath)
print(file_names)
for (i in 1:length(file_names)) {
    print(111)
    path=paste(finalpath,sep = "",file_names[i])
    print(path)
    #csv<- read.csv(file=path, header=TRUE, sep=",")
    csv<- read_excel(path)
    csv<-csv[-1]
    sk <- sk_esd(csv)
    
    
    #plot(sk)
    
    resultpath=paste(skresultpath,sep = "",file_names[i])
    resultpath=paste(resultpath,sep = "",".txt")
    print(resultpath)
    
    write.table (sk[["groups"]], resultpath)
    print(222)
}

