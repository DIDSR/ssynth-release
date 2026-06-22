## This R script run a toy example in preparing input for mrmcAnalaysisOREstimation.R
library(readr)
library(jsonlite)
library(iMRMC)

## set the working directory ####
# This is the path to iMRMCforDice
# setwd("~/GitHub repositories/iMRMCforDice")
source('doIMRMCestOR.R')
source('doIMRMCtestOR.R')

options(readr.show_col_types = FALSE) # ignore readr outputs
## read data ####
# data <- read_csv("/projects01/VICTRE/elena.sizikova/code/breast/iMRMCforDice/data/sample1.csv")
args = commandArgs(trailingOnly=TRUE)
print(args[1])
print(args[2])
data <- read_csv(args[1])

# ## convert matrix mode MRMC data to list mode ####
# readers <- colnames(data)
# cases <- paste0('case', rownames(data))
# nReaders <- length(readers)
# 
# dfMRMC <- data.frame()
# for(i in 1:nReaders){
#   tempDF <- data.frame(readers[i], cases, 'models', data[,i])
#   names(tempDF) = c("readerID","caseID","modalityID","score")
#   dfMRMC <- rbind(dfMRMC,tempDF)
# }
# dfMRMC$readerID <- factor(dfMRMC$readerID)
# dfMRMC$caseID <- factor(dfMRMC$caseID)
# dfMRMC$modalityID <- factor(dfMRMC$modalityID)

## convert matrix mode MRMC data to list mode using convertDF ####
readers <- colnames(data)
data$caseID <- paste0('case', rownames(data))
dfMRMC <- convertDF(data, "matrixMode", "listMode",readers)

dfMRMC$readerID <- factor(dfMRMC$readerID)
dfMRMC$caseID <- factor(dfMRMC$caseID)
dfMRMC$modalityID <- factor("models")
dfMRMC$score <- unlist(dfMRMC$score)

## run doIMRMCestOR ####
result <- doIMRMCestOR(dfMRMC,c("models"))

# result
write_json(result, args[2])

# ## run doIMRMCtestOR ####
# test <- doIMRMCtestOR(result, 0.67)
# print(test)
