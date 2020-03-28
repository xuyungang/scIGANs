## last update: 2020/03/28

args <- commandArgs(T)
job_name = args[1]
tmp = args[2]
outdir = args[3]
timestamp = args[4]
library(data.table)
#file = "ercc.txt"
load(paste(tmp,"/original.RData", sep = ""))
file = paste(tmp,"/scIGANs-", job_name,".csv", sep = "")

#file = "scIGANs-ercc.txt_ercc.label.txt.csv" 
#load("original.RData")
d <- fread(file, header = T)
gcm <- d[,-1]*reads_max_cell

gcm_out <- cbind(Gene_ID = genenames, t(gcm[,1:geneCount]))
colnames(gcm_out) = cellnames
fwrite(gcm_out, file="scIGANs-ercc.txt_ercc.label.txt", col.names = T, row.names = F, sep = "\t", quote = F)

#gcm_out[,-1]<-gcm_out[,-1]*matrix(reads_max_cell,nrow=geneCount,ncol=cellCount,byrow = T)
outfile = paste(outdir,"/scIGANs_",job_name,".txt", sep = "")
fwrite(gcm_out, file=outfile, col.names = T, row.names = F, sep = "\t", quote = F)
message(paste("\nCompleted!!!", Sys.time()))
message(paste("Imputed matrix:", outfile))

