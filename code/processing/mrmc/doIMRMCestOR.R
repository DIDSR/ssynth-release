# doIMRMCestOR ####
#' MRMC analysis by Obuchowski and Rockette ( Obuchowski1995_Commun-Stat-Simulat_v24p285)
#'
#' @description OR's method to estimate the MRMC variance for a given theta.
#' This function requires an nModalities by nReaders covariance matrix to estimate
#' covOR = (cov1, cov2, cov3), as described by Hillis (2014, SIM, Section 2.2)
#'
#' @param dfMRMC dataframe: MRMC dataframe with 4 columns: readerID, caseID, modalityID, score and nReadersxnCasesxnModalities rows
#' @param modalitytoEestimate array: One modality to estimate reader averaged performance or two modalities to compute difference between modalities 
#' @param is.pooled bool: Flat indicating whether or not to do single modality analysis using pooled data
#'
#' @return  list
#' \itemize{
#'   \item  \code{estimation}    : Dataframe for reader-averaged performance for each modality and difference
#'                                 between modalities (if two modalities are compared), including point estimation,
#'                                 the variance of point estimation, and the degree of freedom
#'   \item  \code{theta.hat} [nModalities, nReaders]: Performance estimates for one/two modalities by all readers
#'   \item  \code{cov.hat} [nModalities * nReaders, nModalities * nReaders]: Covariance matrix between each reader x modality}
#'   \item  \code{covOR}     [6] : Components of variance of the Obuchowski and Rockette method
#'   \itemize{
#'     \item                           cov1, cov2(pooled over modalities), cov3,
#'     \item                           varC, cov2(modalityA only), cov2(modalityB only)
#'   }
#'   \item  \code{MSOR}      [3] : Mean squares for modality, reader, and modality-reader interaction
#' }
#'
#' @export
#'
doIMRMCestOR = function(dfMRMC, modalitytoEstimate=c("testA","testB"), is.pooled=T) {
  
  if (length(setdiff(c("readerID", "caseID", 
                       "modalityID", "score"), names(dfMRMC)))) {
    stop("The data frame does not include the key columns: readerID, caseID, modalityID, score.")
  }
  
  df <- droplevels(dfMRMC[dfMRMC$modalityID %in% modalitytoEstimate,])
  
  nModalities <- nlevels(df$modalityID)
  if (nModalities==1){
    is.singleModality <- T
    modalitytoEstimate <- rep(modalitytoEstimate,2)
    nModalities <- 2
  }else if (nModalities==2){
    is.singleModality <- F
  }else{
    stop("The number of modalities to analysis should be either 1 or 2.")
  }
  
  nReaders <- nlevels(df$readerID)
  nCases <- nlevels(df$caseID)
  readers <- levels(df$readerID)
  cases <- levels(df$caseID)
  
  # Convert iMRMC dataframe to matrix form for each modality
  data <- matrix(0,nCases, nReaders*2)
  colnames(data) <- rep(readers,2)
  rownames(data) <- cases
  for(i in 1:2){
    data[,(1:nReaders)+(i-1)*nReaders] <- convertDFtoScoreMatrix(df, modalitytoEstimate[i])
  }

  # get theta.hat 
  theta.hat <- t(array(apply(data,2,mean),list(nReaders,2)))
  colnames(theta.hat) <- readers
  rownames(theta.hat) <- modalitytoEstimate
  
  # get cov.hat 
  if(is.singleModality){
    cov.hat <- kronecker(diag(nModalities),cov(data[,1:nReaders]))
  }else{
    cov.hat <- cov(data)
  }
  colnames(cov.hat) <- paste0(rep(readers,2),'.',
                              rep(modalitytoEstimate,each=nReaders))
  rownames(cov.hat) <- paste0(rep(readers,2),'.',
                              rep(modalitytoEstimate,each=nReaders))

  #Cov.1 and Cov.3
  # subV12 is the off-diagonal submatrix
  subV12 <- cov.hat[(nReaders + 1):(2*nReaders), 1:nReaders]
  Cov.1 <- mean(diag(subV12))
  Cov.3 <- mean(subV12[row(subV12) != col(subV12)])
  
  #Cov2 and VarE from the first-modality data
  # subV11 is the on-diagonal submatrix for modality 1
  subV11 <- cov.hat[1:nReaders,1:nReaders]
  Cov21 <- mean(subV11[row(subV11) != col(subV11)])
  VarE1 <- mean(diag(subV11))
  
  #Cov2 and VarE from the second-modality data
  # subV11 is the on-diagonal submatrix for modality 2
  subV22 <- cov.hat[(nReaders + 1):(2*nReaders), (nReaders + 1):(2*nReaders)]
  Cov22 <- mean(subV22[row(subV22) != col(subV22)])
  VarE2 <- mean(diag(subV22))
  
  #Cov2 and VarE averaged over the two modalities
  Cov.2 <- mean(c(Cov21, Cov22))
  VarE <- mean(c(VarE1, VarE2))
  
  theta.i <- rowMeans(theta.hat, na.rm = TRUE)
  theta.j <- colMeans(theta.hat, na.rm = TRUE)
  theta.d <- mean(theta.j, na.rm = TRUE)
  theta.ii <- matrix(rep(theta.i, times = nReaders),nrow = nModalities)
  theta.jj <- matrix(rep(theta.j, each = nModalities),nrow = nModalities)
  theta.dd <- matrix(theta.d,nModalities,nReaders)
  MS.T <- nReaders*var(theta.i, na.rm = TRUE)
  MS.R <- nModalities*var(theta.j, na.rm = TRUE)
  MS.TR <- sum((theta.hat - theta.ii - theta.jj + theta.dd) ^ 2, na.rm = TRUE)/(nModalities - 1)/(nReaders - 1)
  
  # std err for the difference between 2 elements of theta.hat
  se.dif <- sqrt(2*(MS.TR + max(nReaders*(Cov.2 - Cov.3), 0))/nReaders)
  if(MS.TR == 0){
    df.H <- 0
  }else{
    df.H <- (nModalities - 1)*(nReaders - 1)*(MS.TR + max(nReaders*(Cov.2 - Cov.3),0)) ^ 2/(MS.TR ^ 2)
  }
    
  # inference on a single modality
  # if is.pooled==F then only use modality-specific data
  #   else pool data across modalities
  if (!is.pooled) {
    MSR.i <- c(var(theta.hat[1, ], na.rm = TRUE), var(theta.hat[2, ], na.rm = TRUE))
    Cov.2i <- c(Cov21, Cov22)
    Cov.2i[Cov.2i < 0] <- 0
    MSden.i <- MSR.i + nReaders*Cov.2i
    df.sgl <- (nReaders - 1)*MSden.i ^ 2/MSR.i ^ 2
    se.i <- sqrt(MSden.i/nReaders)
  }else {
    MSR.i <- rep(MS.R,2)
    df.sgl <- (nReaders - 1)*((MS.R + (nModalities - 1)*MS.TR + 
                                 nModalities*nReaders*max(Cov.2,0)) ^ 2)/((MS.R ^ 2) + (nModalities - 1)*(MS.TR ^ 2))
    se.i <- sqrt((MS.R + (nModalities - 1)*MS.TR + nModalities*nReaders*max(Cov.2,0))/nModalities/nReaders) # std err for a single element of theta.hat
  }
  
  covOR <- rep(NA, 12)
  covOR[1] <- (MS.R - MS.TR)/nModalities - Cov.1 + Cov.3
  covOR[2] <- MS.TR - VarE + Cov.1 + Cov.2 - Cov.3
  covOR[3] <- Cov.1;
  covOR[4] <- Cov.2;
  covOR[5] <- Cov.3;
  covOR[6] <- VarE;
  covOR[7] <- MSR.i[1] + Cov21 - VarE1;
  covOR[8] <- MSR.i[2] + Cov22 - VarE2;
  covOR[9] <- Cov21;
  covOR[10] <- Cov22;
  covOR[11] <- VarE1;
  covOR[12] <- VarE2;
  
  names(covOR) <- c("varR", "varTR", "cov1", "cov2", "cov3", "varE",
                    "varR.1", "varR.2", "cov2.1", "cov2.2", "varE.1", "varE.2")
  
  MSOR <- c(MS.T, MS.R, MS.TR)
  names(MSOR) <- c("MS.T", "MS.R", "MS.TR")
  
  estimation <- data.frame(readerAveragedPerformance = theta.i,
                           variance = se.i^2,
                           df = df.sgl)
  if(is.singleModality){
    estimation <- estimation[1,]
    rownames(estimation) <- modalitytoEstimate[1]
  }else{
    estimation[3,] <- data.frame(readerAveragedPerformance = diff(theta.i),
                                 variance = se.dif^2,
                                 df = df.H,
                                 row.names = "modalityDiff")
  }
  
  list(estimation=estimation, theta.hat = theta.hat, cov.hat=cov.hat, covOR = covOR, 
       MSOR = MSOR)
}

