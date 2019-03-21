#library(tictoc)## runtime counter
#library(Rtsne.multicore)
#library(Rtsne)
message("Check package: SamSPECTRAL...")
if (!requireNamespace("BiocManager", quietly = TRUE)){
  install.packages("BiocManager", quiet = T, repos = "https://cran.r-project.org")
  BiocManager::install("SamSPECTRAL", version = "3.8", update = T, ask = F)
}
library(SamSPECTRAL)
message("Check package: readr...")
if(!require("readr", quietly = T)){
  install.packages("readr", quiet = T, dependencies = T,repos = "https://cran.r-project.org")
}
library(readr)
message("Check package: bootSVD...")
if(!require("bootSVD", quietly = T)){
  install.packages("bootSVD", quiet = T, dependencies = T,repos = "https://cran.r-project.org")
}
library(bootSVD)

#if(!require("reticulate", quietly = T)){
#  install.packages("reticulate", quiet = T, dependencies = T,repos = "http://cran.us.r-project.org")
#}
#library(reticulate)

args <- commandArgs(T)
file = args[1]
tmp = args[2]
label = args[3]
ncls = 0 ## the number of clusters
#cluster = args[2]


## randomly sample rows from the matrix to fill the origianl matrix to a desired row count
upSample <- function(matrix, rowNum){
  mRows <- dim(matrix)[1]
  if(mRows>=rowNum){
    return(matrix)
  } else{
    getRows = sample(1:mRows, rowNum-mRows, replace = T)
    return(rbind(matrix,matrix[getRows,]))
  }
  
}

#d<-read_tsv('Merge.readsImpute.transcript.expression.txt')
#file = 'GBMMerge.readsImpute.gene.expression.NoZero.txt'
#file = "ercc.txt"
#basename(file)
if(!file.exists(tmp)) dir.create(tmp)
if(is.null(file) || is.na(file)){
  stop("The tab-delimited file for expression matrix is required!!!") 
}
d<-read_tsv(file)
genenames = c(d[,1])[[1]]
cellnames = colnames(d)
d <- d[, -1]
geneCount<-dim(d)[1] ## gene count
cellCount<-dim(d)[2] ## cell count

## check the number of rows (gene numbers)
fig_h<-124 ##124^=15376
numD <- ceiling(geneCount/fig_h^2)

gcm <- upSample(d, numD*fig_h^2)

#normalize data such that maximum vale for each cell equals to 1
reads_max_cell<-apply(gcm,2,max,na.rm=T)## the max value of each column
#write.csv(reads_max_cell,paste("tmp/",basename(file), ".max.csv", sep = ""),quote=F,row.names = T)
save(genenames, cellnames, geneCount, cellCount, reads_max_cell, numD, file = paste(tmp,"/original.RData", sep = ""))
gcm_n<-gcm/matrix(reads_max_cell,nrow=geneCount,ncol=cellCount,byrow = T)
set.seed(100)
#process the label
if(is.null(label) || is.na(label)){## if no label file provided, then run pre-cluster to generate cluster label for each cell
    ##do PCA
  
  if(!require("Rtsne", quietly = T)){
    install.packages("Rtsne", quiet = T, dependencies = T,repos = "https://cran.r-project.org")
  }
  library(Rtsne)
    Ys<-scale(t(gcm_n))
    ysvd<-fastSVD(Ys)
    pcsn<-ysvd$u*matrix(ysvd$d,nrow=dim(ysvd$u)[1],ncol=dim(ysvd$u)[2],byrow=T) ##(dim = cell*100)
    #full<-pcsn[,1:50]
    tsne3 <- Rtsne(pcsn, dims = 3, theta=0.2, perplexity=30, verbose=TRUE, max_iter = 1000)
    full<-tsne3$Y
    normal.sigma <-50

    m<-SamSPECTRAL(full,separation.factor =0.5,normal.sigma = normal.sigma)
   ncls = length(unique(m))
    #output
    cluster<-data.frame(cls=m[!is.na(m)])
    label = paste(file,".label.csv", sep = "")
    write.csv(cluster,label,quote=F,row.names = T)
}else{## convert the provided labels to integers
  cls = read_tsv(label, col_names = F)
  cls.factor = factor(cls$X1)
  ncls = length(levels(cls.factor))
  cluster<-data.frame(cls = unclass(cls.factor))
  label = paste(file,".label.csv", sep = "")
  write.csv(cluster,label,quote=F,row.names = T)
}



for(i in 1:numD){
  #i=1
  start = (1+(i-1)*fig_h^2)
  end = i*fig_h^2
  rows = c( start:end )
  sub_gcm <- gcm_n[rows,]
 
  subd<-as.data.frame(sub_gcm)
  subfile = paste(tmp,"/",i,"_",basename(file), sep = "")
  write.csv(subd[,!is.na(m)],subfile,quote=F,row.names = T)
}
cat(paste("_",numD,"_",ncls,sep = ""))
