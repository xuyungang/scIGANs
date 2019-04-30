## last update: 2019/04/28
args <- commandArgs(T)
file = args[1]
tmp = args[2]
label = args[4]
logfile = args[3]
ncls = 0 ## the number of clusters
message("Check dependent packages...")
write(paste(date(), "\tCheck dependent packages...", sep=""), logfile, append = T)
library(SamSPECTRAL)
library(Rtsne)
message("Done!")
write(paste(date(),"\tDone!", sep=""), logfile, append = T)


## randomly sample rows from the matrix to fill the origianl matrix to a desired row count
upSample_random <- function(matrix, rowNum){
  mRows <- dim(matrix)[1]
  if(mRows>=rowNum){
    return(matrix)
  } else{
    getRows = sample(1:mRows, rowNum-mRows, replace = T)
    return(rbind(matrix,matrix[getRows,]))
  }
  
}
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

if(!file.exists(tmp)) dir.create(tmp)
if(is.null(file) || is.na(file)){
  write("ERROR:The tab-delimited file for expression matrix is required!!!", logfile, append = T)
  stop("ERROR:The tab-delimited file for expression matrix is required!!!")
}
d<- read.table(file, header = T, sep = "\t")
genenames = as.character(d[,1])
cellnames = colnames(d)
d <- d[, -1]
geneCount<-dim(d)[1] ## gene count
cellCount<-dim(d)[2] ## cell count

## check the number of rows (gene numbers)
#fig_h<-124 ##124^=15376
#numD <- ceiling(geneCount/fig_h^2)

#gcm <- upSample_zero(d, numD*fig_h^2)
## upSample the matrix to rows that the sqrt of the number is >= geneCount
#numD = 1
fig_h = ceiling(sqrt(geneCount))
gcm <- upSample_zero(d, fig_h^2)

#normalize data such that maximum vale for each cell equals to 1
reads_max_cell<-apply(gcm,2,max,na.rm=T)## the max value of each column
save(genenames, cellnames, geneCount, cellCount, reads_max_cell, file = paste(tmp,"/original.RData", sep = ""))
gcm_n<-gcm/matrix(reads_max_cell,nrow=fig_h^2,ncol=cellCount,byrow = T)
set.seed(100)
#process the label

if(is.null(label) || is.na(label)){## if no label file provided, then run pre-cluster to generate cluster label for each cell
    message("No label file provided, generating labels...")
    write(paste(date(), "\tNo label file provided, generating labels...", sep=""), logfile, append = T)
    pcsn <- prcomp(t(gcm_n))
    #full<-pcsn[,1:50]
    tsne3 <- Rtsne(pcsn$x, dims = 3, theta=0.2, perplexity=30, verbose=TRUE, max_iter = 1000)
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


tmpfile = paste(tmp,"/",basename(file), sep = "")
write.csv(gcm_n,tmpfile,quote=F,row.names = T)
cat(paste("|",fig_h,"|",ncls,"|",label,sep = ""))
