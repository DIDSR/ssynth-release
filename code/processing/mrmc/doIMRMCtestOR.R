# doIMRMCtestOR ####
#' MRMC analysis by Obuchowski and Rockette ( Obuchowski1995_Commun-Stat-Simulat_v24p285)
#'
#' @description OR's method to test on the modality effect.
#' This function requires the result from doIMRMCestOR.R to do the hypothesis test
#'
#' @param doIMRMCestOR.result list: output of the function doIMRMCestOR
#' @param mu0 a number indicating the true value of the mean (or difference in means if you are performing a two sample test).
#' @param alpha a number indicating the confidence level.
#'
#' @return  A list with class "\code{htest}" containing the following components
#' \itemize{
#'   \item  \code{statistic}    : the value of the t-statistic.
#'   \item  \code{parameter}    : the degrees of freedom from the t-statistic.
#'   \item  \code{p.value}      : the p-value for the test.
#'   \item  \code{alternative}  : a character string describing the alternative hypothesis.
#'   \item  \code{method}       : a character string indicating what type of t-test was performed
#'   \item  \code{data.name}    : a character string showing the reading modalities
#' }
#'
#' @export
#'
doIMRMCtestOR = function(doIMRMCestOR.result, mu0=NULL, alpha=0.05) {
  
  n <- nrow(doIMRMCestOR.result$estimation)
  estimation <- doIMRMCestOR.result$estimation[n,]
  
  mu <- estimation$readerAveragedPerformance
  sigma <- sqrt(estimation$variance)
  df <- estimation$df
  names(df) <- "df"
  avg <- doIMRMCestOR.result$estimation[,1]
  names(avg) <- paste0("mean for ", rownames(doIMRMCestOR.result$estimation))  
  
  # test for individual modality or difference between two modalities
  if (n==1){
    if(is.null(mu0)){
      stop("Please input the mu0 for mean under the null hypothesis")
    }

    testname <- "One Sample t-test"
    dataname <- rownames(estimation)
    alternativeH <- paste0("true mean for ",rownames(estimation)," is not equal to ",mu0)
  }else{
    if(is.null(mu0)){
      mu0 <- 0
    }
    testname <- "Two Sample t-test"
    groupA <- rownames(doIMRMCestOR.result$estimation)[1]
    groupB <- rownames(doIMRMCestOR.result$estimation)[2]
    dataname <- paste0(groupA," vs ", groupB)
    alternativeH <- paste0("true difference in means between ",groupA,
                          " and ",groupB," is not equal to ",mu0)
  }

  # t-test statistic
  t <- (mu-mu0)/sigma
  names(t) <- "t"
  
  # compute p-value for two-sided test
  p <- 1-pt(-abs(t),df)*2
  
  # confidence interval
  ci <- c(mu-mu0 - qt(1-alpha/2, df)*sigma, mu-mu0 + qt(1-alpha/2, df)*sigma)
  attr(ci, 'conf.level') <- 1-alpha
  
  test <- list(method = testname,
               data.name = dataname,
               statistic = t,
               parameter = df,
               p.value = p,
               alternative = alternativeH,
               estimate = avg,
               conf.int = ci)
  class(test) <- "htest"
  return(test)
  
}

