#------Packages yang Digunakan-----
library(readxl)
library(survival)
library(ggplot2)

#------Input Data-----
diabet <- read_excel("C:/Users/Asus/Documents/Data_MDWH.xlsx")

#------Model Awal-----
diabet.reg <- coxph(Surv(time,status)~gen+usia+diet+olra+bb,diabet)
summary(diabet.reg)

#------Seleksi Model Terbaik dengan -2 Log Likelihood-----
d <- coxph(Surv(time,status)~gen,diabet)
-2*d$loglik
d1 <- coxph(Surv(time,status)~usia,diabet)
-2*d1$loglik
d2 <- coxph(Surv(time,status)~diet,diabet)
-2*d2$loglik
d3 <- coxph(Surv(time,status)~olra,diabet)
-2*d3$loglik
d4 <- coxph(Surv(time,status)~bb,diabet)
-2*d4$loglik
d5 <- coxph(Surv(time,status)~usia+gen,diabet)
-2*d5$loglik
d6<- coxph(Surv(time,status)~usia+diet,diabet)
-2*d6$loglik
d7 <- coxph(Surv(time,status)~usia+olra,diabet)
-2*d7$loglik
d8 <- coxph(Surv(time,status)~usia+bb,diabet)
-2*d8$loglik
d9 <- coxph(Surv(time,status)~usia+gen+diet,diabet)
-2*d9$loglik
d10 <- coxph(Surv(time,status)~usia+gen+olra,diabet)
-2*d10$loglik
d11 <- coxph(Surv(time,status)~usia+gen+bb,diabet)
-2*d11$loglik
d12 <- coxph(Surv(time,status)~usia+gen+diet+olra,diabet)
-2*d12$loglik
d13 <- coxph(Surv(time,status)~usia+gen+diet+bb,diabet)
-2*d13$loglik

#------Model Terbaik-----
diabet.reg2 <- coxph(Surv(time,status)~usia+gen+diet,diabet)
summary(diabet.reg2)

