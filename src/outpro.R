
args <- commandArgs(T)
file = args[1]
tmp = args[2]
outdir = args[3]

#file = "ercc.txt"
load(paste(tmp,"/original.RData", sep = ""))
gcm <- data.frame()
for(i in 1:numD){
  i = 1
  d <- read.csv(file = paste(tmp,"/scIGANs-",i,"_", file,".csv", sep = ""), header = T)
  gcm = rbind(gcm, t(d[,-1]))
}

gcm <- cbind("Gene_ID"=NA, gcm)
colnames(gcm) = cellnames
gcm_out = gcm[c(1:geneCount),]
gcm_out[,1] = genenames

gcm_out[,-1]<-gcm_out[,-1]*matrix(reads_max_cell,nrow=geneCount,ncol=cellCount,byrow = T)
outfile = paste(outdir,"/scIGANs_",format(Sys.time(), "%Y%m%d%H%M"),"_",file, sep = "")
write.table(gcm_out, file=outfile, col.names = T, row.names = F, sep = "\t", quote = F)
message(paste("Completed!!!", Sys.time()))
message(paste("Imputed matrix:", outfile))

