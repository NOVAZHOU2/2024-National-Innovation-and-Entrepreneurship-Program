library(ScottKnottESD)
library(readxl)

# 文件路径
finalpath = 'E:\\Desktop\\Github desktop备份集\\Different-Unsupervised-Algorithms-for-Software-Defect-Prediction\\CXX_CLUSTRTING\\Clustering4UEADP-main\\RQ1\\DrawPicData\\'
skresultpath = 'E:\\Desktop\\Github desktop备份集\\Different-Unsupervised-Algorithms-for-Software-Defect-Prediction\\CXX_CLUSTRTING\\Clustering4UEADP-main\\RQ1\\pictures\\'
setwd(skresultpath)

file_names <- list.files(finalpath, pattern = "^[^~].*\\.xlsx$")

for (i in 1:length(file_names)) {
  if (grepl(pattern = ".xlsx$", x = file_names[i])) {
    #print(file_names[i])

    # 去除文件名后缀
    name <- strsplit(file_names[i], split = ".xlsx")[[1]][1]
    #print(name)

    # 文件路径
    path <- paste(finalpath, sep = "", file_names[i])
    #print(path)

    csv <- read_excel(path)
    csv <- csv[-1]

    # 将列名去除“.及后缀”
    colnames(csv) <- gsub("\\..*", "", colnames(csv))

    sk <- sk_esd(csv)
    print(sk)
    #print(sk$statistics$mean)
    future <- paste('RQ1_Ranking_', name, sep = "", ".jpg")
    y_axis_data <- data.frame(Method = rownames(sk$statistics), Mean = sk$statistics$mean)
    output_csv_path <- paste(skresultpath, name, "_y_axis_data.csv", sep = "")
    write.csv(y_axis_data, output_csv_path, row.names = FALSE)# 保存图片
    jpeg(file = future, width = 5000, height = 2000, units = 'px', res = 600)
    par(mar = c(8, 4, 2, 1))  # 增大底部边距，以便更好地显示 x 轴标签
    plot(sk,
         mar = c(4, 1, 1, 1),
         las = 2,
         cex.lab = 1.2,
         xlab = '',
         ylab = 'Rankings',
         family = "serif",
         title = NULL,
         #mgp = c(3, -0.5, 0)
         #mgp = c(3, 2, 0),
    )
    dev.off()

    # 保存结果
    resultpath <- paste(skresultpath, name, ".txt", sep = "")
    print(resultpath)
  }
}
