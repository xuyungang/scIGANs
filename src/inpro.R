## last update: 2020/03/28
args <- commandArgs(T)
file = args[1]
tmp = args[2]
logfile = args[3]
label = args[4]
ncls = 0 ## the number of clusters
options("digits" = 4)
message("Check dependent packages...")
write(paste(date(), "\tCheck dependent packages...", sep=""), logfile, append = T)
library(SamSPECTRAL)
library(Rtsne)
library(data.table)
message("Done!")
write(paste(date(),"\tDone!", sep=""), logfile, append = T)


## (DEPRECATED) randomly sample rows from the matrix to fill the origianl matrix to a desired row count
# upSample_random <- function(matrix, rowNum){
#   mRows <- dim(matrix)[1]
#   if(mRows>=rowNum){
#     return(matrix)
#   } else{
#     getRows = sample(1:mRows, rowNum-mRows, replace = T)
#     return(rbind(matrix,matrix[getRows,]))
#   }
#   
# }

## fill the origianl matrix to a desired row count by zeros
upSample_zero <- function(mtx, rowNum){
  mRows <- dim(mtx)[1]
  mCols <- dim(mtx)[2]
  if(mRows>=rowNum){
    return(mtx)
  } else{
    zero_matrix = matrix(rep(0, mCols*(rowNum-mRows)),rowNum-mRows, mCols)
	colnames(zero_matrix) = colnames(mtx)
    return(rbind(mtx,zero_matrix))
  }
  
}
## test only-------------------------------
# file = "ercc.txt"
# tmp = "test.tmp"
# label = "ercc.label.txt"
# logfile = "test.log"
## ---------------------------------------
if(!file.exists(tmp)) dir.create(tmp, showWarnings = F)
if(is.null(file) || is.na(file)){
  write("ERROR:The tab-delimited file for expression matrix is required!!!", logfile, append = T)
  stop("ERROR:The tab-delimited file for expression matrix is required!!!")
}


d<- fread(file, header = T, sep = "\t")
genenames = d[[1]]
cellnames = colnames(d)
d <- d[, -1]
geneCount<-dim(d)[1] ## gene count
cellCount<-dim(d)[2] ## cell count

## upSample the matrix to rows that the sqrt of the number is >= geneCount
#numD = 1
fig_h = ceiling(sqrt(geneCount))
gcm <- upSample_zero(d, fig_h^2)

#normalize data such that maximum vale for each cell equals to 1
reads_max_cell<-apply(gcm,2,max,na.rm=T)## the max value of each column
save(genenames, cellnames, geneCount, cellCount, reads_max_cell, file = paste(tmp,"/original.RData", sep = ""))
gcm_n <- t(gcm)/reads_max_cell
set.seed(100)
#process the label

if(is.null(label) || is.na(label)){## if no label file provided, then run pre-cluster to generate cluster label for each cell
    message("No label file provided, generating labels...")
    write(paste(date(), "\tNo label file provided, generating labels...", sep=""), logfile, append = T)
    pcsn <- prcomp(gcm_n)
    #full<-pcsn[,1:50]
    tsne3 <- Rtsne(pcsn$x, dims = 3, theta=0.2, perplexity=cellCount%/%4, verbose=TRUE, max_iter = 1000)
    full<-tsne3$Y
    normal.sigma <-50

    m<-SamSPECTRAL(full,separation.factor =0.5,normal.sigma = normal.sigma, talk = F)
   ncls = length(unique(m))
    #output
    cluster<-data.frame(Label = m)
    label = paste(file,".label.csv", sep = "")
    write.table(cluster,label,quote=F,row.names = F,col.names = F)
  message("Done!\nLabel was output in ", label,": ", ncls, " clusters")
  write(paste(date(), "\tDone! Label was output in ", label,": ", ncls, " clusters", sep = ""), logfile, append = T)

}else{## convert the provided labels to integers
  message("Label file ", label, " was provided.")
  write(paste(date(), "\tLabel file ", label, " was provided.", sep=""), logfile, append = T)
  cls = read.table(label, header = F, sep = "\t")#read_tsv(label, col_names = F)
  cls.factor = factor(cls[,1])
  ncls = length(levels(cls.factor))
}
write(fig_h, paste(tmp,"/args",sep = ""), append =F)
write(ncls, paste(tmp,"/args",sep = ""), append =T)
write(label, paste(tmp,"/args",sep = ""), append =T)

tmpfile = paste(tmp,"/",basename(file), sep = "")
fwrite(as.data.table(t(gcm_n)) ,tmpfile,quote=F,row.names = T)

