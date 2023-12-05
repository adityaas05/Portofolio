library(readxl)
library(ROSE)
library(smotefamily)
library(DMwR)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(corrplot)
library(reshape2)
library(psych)
library(e1071)
library(glmnet)
library(Metrics) 
library(rpart)
library(caret)
library(pROC)
library(ROCR)

training <- read_excel("C:/Users/Asus/Downloads/Train_ETS.xlsx")
testing <- read_excel("C:/Users/Asus/Downloads/Test_ETS.xlsx")

#-----Penanganan Imbalance Data-----
rose = ROSE(Class~.,data=training)
rose = rose$data
table(rose$Class)

#-------EDA-------
# Menghitung persentase transaksi yang bukan penipuan (non-frauds) dan transaksi penipuan (frauds)
percentage_data_train <- training %>%
  group_by(Class) %>%
  summarise(Count = n()) %>%
  mutate(Percentage = (Count / sum(Count)) * 100)

ggplot(percentage_data_train, aes(x = factor(Class), y = Percentage, fill = factor(Class))) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(Percentage, 2), "%")), vjust = -0.5, size = 4) +
  labs(
    title = "Class Distributions \n (0: No Fraud || 1: Fraud)",
    x = "Class",
    y = "Percentage"
  ) +
  scale_fill_manual(values = c("0" = "blue", "1" = "red")) +
  theme_minimal()

percentage_data_rose <- rose %>%
  group_by(Class) %>%
  summarise(Count = n()) %>%
  mutate(Percentage = (Count / sum(Count)) * 100)

ggplot(percentage_data_rose, aes(x = factor(Class), y = Percentage, fill = factor(Class))) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(Percentage, 2), "%")), vjust = -0.5, size = 4) +
  labs(
    title = "Class Distributions \n (0: No Fraud || 1: Fraud)",
    x = "Class",
    y = "Percentage"
  ) +
  scale_fill_manual(values = c("0" = "blue", "1" = "red")) +
  theme_minimal()

# Membuat dua subplot 
par(mfrow = c(1, 2))
options(repr.plot.width = 18, repr.plot.height = 4)
amount_val <- training$Amount
time_val <- training$Time

hist(amount_val, col = 'red', main = 'Distribution Transaction Amount', xlab = 'Amount', xlim = c(min(amount_val), max(amount_val)))
hist(time_val, col = 'blue', main = 'Distribution Transaction Time', xlab = 'Time', xlim = c(min(time_val), max(time_val)))

bins <- 50
p1 <- ggplot(training[training$Class == 1, ], aes(x = Amount)) +
  geom_histogram(fill = 'blue', color = 'black', bins = bins) +
  labs(title = 'Fraud', x = 'Amount ($)', y = 'Number of Transactions') +
  xlim(0, 20000) +
  scale_y_log10()
p2 <- ggplot(training[training$Class == 0, ], aes(x = Amount)) +
  geom_histogram(fill = 'blue', color = 'black', bins = bins) +
  labs(title = 'Non-Fraud', x = 'Amount ($)', y = 'Number of Transactions') +
  xlim(0, 20000) +
  scale_y_log10()
grid.arrange(p1, p2, ncol = 2)

# Menghilangkan kolom 'No.'
data_train_no_No <- training[, !(names(training) %in% "No.")]
data_rose_no_No <- rose[, !(names(rose) %in% "No.")]

# Matriks korelasi
correlation_matrix <- cor(data_train_no_No)
correlation_df <- melt(correlation_matrix)
ggplot(data = correlation_df, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Korelasi Data Imbalance") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

correlation_matrix <- cor(data_rose_no_No)
correlation_df <- melt(correlation_matrix)
ggplot(data = correlation_df, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Korelasi Data Balance") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#-------Prediksi Jumlah Transaksi-------
set.seed(42)
index_train <- sample(1:nrow(training), 0.8 * nrow(training))
index_test <- setdiff(1:nrow(training), index_train)
data_train <- training[index_train, ]
data_test <- training[index_test, ]

y_train_pred <- data_train$Amount
x_train_pred <- data_train[, !(names(data_train) %in% c("Amount", "No.","Class"))]
y_test_pred <- data_test$Amount
x_test_pred <- data_test[, !(names(data_test) %in% c("Amount", "No.","Class"))]

x_testing <- testing[, !(names(testing) %in% c("Amount", "No.","Class"))]
y_testing <- testing$Amount

# Regresi linear
model_formula <- y_train_pred ~ .
model_lr <- train(
  x = x_train_pred,
  y = y_train_pred,
  method = "lm",
  trControl = trainControl(method = "cv", number = 5),
  verbose = FALSE
)
cv_lr <- train(
  model_formula,
  data = cbind(y_train_pred, x_train_pred),
  method = "lm",
  trControl = trainControl(method = "cv", number = 5),
  verbose = FALSE
)
print(cv_lr)

summary(model_lr)

# Support Vector Regression
model_formula <- y_train_pred ~ .
model_svr <- train(
  x = x_train_pred,
  y = y_train_pred,
  method = "svmLinear",
  trControl = trainControl(method = "cv", number = 5),  # 5-fold cross-validation
  verbose = FALSE
)
cv_svr <- train(
  model_formula,
  data = cbind(y_train_pred, x_train_pred),
  method = "svmLinear",
  trControl = trainControl(method = "cv", number = 5),  # 5-fold cross-validation
  verbose = FALSE
)
print(cv_svr)

# Lasso Regressor
x_train_mat <- as.matrix(x_train_pred)
y_train_vec <- as.vector(y_train_pred)

model_formula <- as.formula(paste("y_train_pred ~ ."))
lambda_seq <- 10^seq(10, -2, length = 100)
lasso_cv <- cv.glmnet(x_train_mat, y_train_vec, alpha = 1, lambda = lambda_seq, nfolds = 5)
best_lambda <- lasso_cv$lambda.min

lasso_model <- glmnet(x_train_mat, y_train_vec, alpha = 1, lambda = best_lambda)
x_train_mat <- as.matrix(x_train_pred)
y_train_lasso <- predict(lasso_model, newx = x_train_mat, s = best_lambda)

rmse_value <- rmse(y_train_pred, y_train_lasso)
rsquared_value <- cor(y_train_lasso, as.vector(y_train_pred))^2
mae_value <- mean(abs(y_train_lasso - as.vector(y_train_pred)))
cat("RMSE:", rmse_value, "\n")
cat("R-squared:", rsquared_value, "\n")
cat("MAE:", mae_value, "\n")

coefs <- coef(lasso_model)
coefs_matrix <- as.matrix(feature_importances)
coef_table <- data.frame(Variable = rownames(coefs_matrix), Coefficient = coefs_matrix)
sorted_coef_table <- coef_table %>%
  arrange(desc(`s0`))
print(sorted_coef_table)
sorted_coef_table <- tail(sorted_coef_table, -1)
top_10_coef <- head(sorted_coef_table, 10)
top_10_coef$s0 <- as.numeric(top_10_coef$s0)
ggplot(top_10_coef, aes(x = s0, y = reorder(Variable, s0), fill = s0)) +
  geom_col() +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Feature Important", x = "Koefisien", y = "Variabel") +
  theme_minimal()

# Lasso Regression Data Test Check
x_testing <- testing[, !(names(testing) %in% c("Amount", "No.","Class"))]
y_testing <- testing$Amount
x_testing_mat <- as.matrix(x_testing)
y_testing_vec <- as.vector(y_testing)
y_testing_lasso <- predict(lasso_model, newx = x_testing_mat, s = best_lambda)
rmse_value_test <- rmse(y_testing_vec, y_testing_lasso)
rsquared_value_test <- cor(y_testing_lasso, as.vector(y_testing_vec))^2
mae_value_test <- mean(abs(y_testing_lasso - as.vector(y_testing_vec)))
cat("RMSE:", rmse_value_test, "\n")
cat("R-squared:", rsquared_value_test, "\n")
cat("MAE:", mae_value_test, "\n")

# File Prediksi 
result_test_prediksi <- data.frame(Predicted = y_testing_lasso, Actual = testing$Amount)
write.csv(result_test_prediksi, file = "2043201082_prediksi.csv", row.names = FALSE)

#-------Kesimpulan Model Prediksi Validation-------
model_pred <- c("Linear Regression", "Support Vector Regression", "Lasso Regressor")
rmse_pred <- c(
  rmse(y_test_pred, predict(cv_lr, newdata = x_test_pred)),
  rmse(y_test_pred, predict(cv_svr, newdata = x_test_pred)),
  rmse_value
)
rsquared_pred <- c(
  cor(predict(cv_lr, newdata = x_test_pred), as.vector(y_test_pred))^2,
  cor(predict(cv_svr, newdata = x_test_pred), as.vector(y_test_pred))^2,
  rsquared_value
)
mae_pred <- c(
  mean(abs(predict(cv_lr, newdata = x_test_pred) - as.vector(y_test_pred))),
  mean(abs(predict(cv_svr, newdata = x_test_pred) - as.vector(y_test_pred))),
  mae_value
)
evaluation_table_pred <- data.frame(Model = model_pred, RMSE = rmse_pred, R_squared = rsquared_pred, MAE = mae_pred)
print(evaluation_table_pred)

#-------Klasifikasi Penipuan-------
# Logistic Regression
model_logit <- function() {
  formula <- as.formula(paste("Class ~ ", paste(setdiff(names(rose), c("No.", "Class")), collapse = " + ")))
  roselogit <- glm(formula, data = rose, family = "binomial")
  return(roselogit)
}
roselogit <- model_logit()
summary(roselogit)
rosepredict = predict(roselogit, type = 'response')
rosepredict_logit = ifelse(rosepredict >= 0.5, 1.0, 0.0)

conf_matrix_logit <- confusionMatrix(as.factor(rose$Class), as.factor(rosepredict_logit), positive = "1")
f1_score_logit <- conf_matrix_logit$byClass["F1"]
print(conf_matrix_logit)
cat("F1-score:", f1_score_logit, "\n")

# Menampilkan Heatmap CM Logit
conf_matrix <- conf_matrix_logit$table
print(conf_matrix)
heatmap_plot <- ggplot(data = as.data.frame(conf_matrix), aes(x = Reference, y = factor(Prediction, levels = c("1", "0")), fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1, size = 10) +  # Increase font size
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix Heatmap", x = "Reference", y = "Prediction") +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15))
print(heatmap_plot)

# SVM
model_svm <- function() {
  formula <- as.formula(paste("Class ~ ", paste(setdiff(names(rose), c("No.", "Class")), collapse = " + ")))
  rosesvm <- svm(formula, data = rose, kernel = "radial", probability = TRUE)
  return(rosesvm)
}

rosesvm <- model_svm()
rosepredict <- predict(rosesvm, rose, probability = TRUE)
rosepredict_svm <- as.factor(ifelse(rosepredict >= 0.5, 1, 0))
levels(rosepredict_svm) <- levels(rose$Class)

conf_matrix_svm <- confusionMatrix(rosepredict_svm, as.factor(rose$Class), positive = "1")
f1_score_svm <- conf_matrix_svm$byClass["F1"]
print(conf_matrix_svm)
cat("F1-score:", f1_score_svm, "\n")

# DT
model_decision_tree <- function() {
  formula <- as.formula(paste("Class ~ ", paste(setdiff(names(rose), c("No.", "Class")), collapse = " + ")))
  rosedt <- rpart(formula, data = rose, method = "class")
  return(rosedt)
}

rosedt <- model_decision_tree()
rosepredict_dt <- predict(rosedt, type = "class")

conf_matrix_dt <- confusionMatrix(as.factor(rose$Class), rosepredict_dt, positive = "1")
f1_score_dt <- conf_matrix_dt$byClass["F1"]
print(conf_matrix_dt)
cat("F1-score:", f1_score_dt, "\n")

#NBC
model_naive_bayes <- function() {
  formula <- as.formula(paste("Class ~ ", paste(setdiff(names(rose), c("No.", "Class")), collapse = " + ")))
  rosenb <- naiveBayes(formula, data = rose)
  return(rosenb)
}

rosenb <- model_naive_bayes()
rosepredict_nb <- predict(rosenb, newdata = rose)

conf_matrix_nb <- confusionMatrix(rosepredict_nb, as.factor(rose$Class), positive = "1")
f1_score_nb <- conf_matrix_nb$byClass["F1"]
print(conf_matrix_nb)
cat("F1-score:", f1_score_nb, "\n")

# ROC Curve
predictions_roc <- data.frame(
  LogisticRegression = rosepredict_logit,
  SVM = rosepredict_svm,
  DecisionTree = rosepredict_dt,
  NaiveBayes = rosepredict_nb
)
roc_objs <- lapply(colnames(predictions_roc), function(model_name) {
  prediction_obj <- prediction(predictions_roc[[model_name]], rose$Class)
  performance_obj <- performance(prediction_obj, "tpr", "fpr")
  auc <- performance(prediction_obj, "auc")@y.values[[1]]
  list(model_name = model_name, performance_obj = performance_obj, auc = auc)
})
colors <- rainbow(length(roc_objs))
plot(NULL, xlim = c(0, 1), ylim = c(0, 1), xlab = "False Positive Rate", ylab = "True Positive Rate", main = "ROC Curve")
for (i in 1:length(roc_objs)) {
  plot(roc_objs[[i]]$performance_obj, col = colors[i], add = TRUE)
}
legend("bottomright", legend = sapply(roc_objs, function(x) paste(x$model_name, "AUC =", sprintf("%.4f", x$auc))), col = colors, lty = 1)

rosepredict_new <- predict(roselogit, newdata = testing, type = 'response')
rosepredict_new <- ifelse(rosepredict_new >= 0.5, 1.0, 0.0)
conf_matrix_logit_test <- confusionMatrix(as.factor(testing$Class), as.factor(rosepredict_new), positive = "1")
f1_score_logit_test <- conf_matrix_logit_test$byClass["F1"]
conf_matrix_logit_test
cat("F1-score:", f1_score_logit_test, "\n")

conf_matrix_test <- conf_matrix_logit_test$table
print(conf_matrix_test)
heatmap_plot_test <- ggplot(data = as.data.frame(conf_matrix_test), aes(x = Reference, y = factor(Prediction, levels = c("1", "0")), fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1, size = 10) +  # Increase font size
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix Heatmap", x = "Reference", y = "Prediction") +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15))
print(heatmap_plot_test)

roc_curve_logit_test <- roc(as.factor(testing$Class), rosepredict_new)
plot(roc_curve_logit_test, main = "Kurva ROC - Logistic Regression (Test)", auc.polygon = TRUE, print.auc = TRUE, grid = TRUE)

# File Prediksi Klasifikasi
result_test_klasifikasi <- data.frame(Predicted = rosepredict_new, Actual = testing$Class)
write.csv(result_test_klasifikasi, file = "2043201082_klasifikasi.csv", row.names = FALSE)

#-------Kesimpulan Model Klasifikasi-------
models <- c("Logistic Regression", "SVM", "Decision Tree", "Naive Bayes")

accuracy <- c(
  conf_matrix_logit$overall["Accuracy"],
  conf_matrix_svm$overall["Accuracy"],
  conf_matrix_dt$overall["Accuracy"],
  conf_matrix_nb$overall["Accuracy"]
)
sensitivity <- c(
  conf_matrix_logit$byClass["Sensitivity"],
  conf_matrix_svm$byClass["Sensitivity"],
  conf_matrix_dt$byClass["Sensitivity"],
  conf_matrix_nb$byClass["Sensitivity"]
)
specificity <- c(
  conf_matrix_logit$byClass["Specificity"],
  conf_matrix_svm$byClass["Specificity"],
  conf_matrix_dt$byClass["Specificity"],
  conf_matrix_nb$byClass["Specificity"]
)
f1_score <- c(f1_score_logit, f1_score_svm, f1_score_dt, f1_score_nb)

summary_table <- data.frame(
  Model = models,
  Accuracy = accuracy,
  Sensitivity = sensitivity,
  Specificity = specificity,
  F1_Score = f1_score
)
print(summary_table)
