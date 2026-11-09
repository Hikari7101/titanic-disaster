data_dir <- Sys.getenv("DATA_DIR", "/app/data")

load_csv <- function(path) {
  if (requireNamespace("readr", quietly = TRUE)) {
    as.data.frame(readr::read_csv(path, show_col_types = FALSE))
  } else {
    read.csv(path, stringsAsFactors = FALSE)
  }
}

cat(strrep("=", 60), "\n")
cat("Titanic Survival Prediction - R\n")
cat(strrep("=", 60), "\n")

# ---------- 1. Load training data ----------
train <- load_csv(file.path(data_dir, "train.csv"))
cat(sprintf("\n[1] Loaded train.csv: %d rows, %d columns\n", nrow(train), ncol(train)))
print(head(train[, c("PassengerId", "Survived", "Pclass", "Sex", "Age", "Fare", "Embarked")]))

# treat empty strings as missing
train$Embarked[!is.na(train$Embarked) & train$Embarked == ""] <- NA

# ---------- 2. Explore ----------
cat("\n[2] Missing values in train.csv:\n")
print(colSums(is.na(train)))
cat(sprintf("\nOverall survival rate: %.3f\n", mean(train$Survived)))
cat("\nSurvival rate by Sex:\n")
print(tapply(train$Survived, train$Sex, mean))
cat("\nSurvival rate by Pclass:\n")
print(tapply(train$Survived, train$Pclass, mean))

# ---------- 3. Clean / adjust ----------
age_median <- median(train$Age, na.rm = TRUE)
fare_median <- median(train$Fare, na.rm = TRUE)
embarked_mode <- names(sort(table(train$Embarked), decreasing = TRUE))[1]

prepare <- function(df, name) {
  cat(sprintf("\n[3] Preparing %s:\n", name))
  df$Embarked[!is.na(df$Embarked) & df$Embarked == ""] <- NA

  n <- sum(is.na(df$Age))
  df$Age[is.na(df$Age)] <- age_median
  cat(sprintf("  - Filled %d missing Age values with train median (%.1f)\n", n, age_median))

  n <- sum(is.na(df$Fare))
  df$Fare[is.na(df$Fare)] <- fare_median
  cat(sprintf("  - Filled %d missing Fare values with train median (%.2f)\n", n, fare_median))

  n <- sum(is.na(df$Embarked))
  df$Embarked[is.na(df$Embarked)] <- embarked_mode
  cat(sprintf("  - Filled %d missing Embarked values with train mode ('%s')\n", n, embarked_mode))

  df$Sex <- ifelse(df$Sex == "female", 1, 0)
  cat("  - Encoded Sex: male=0, female=1\n")

  df$FamilySize <- df$SibSp + df$Parch + 1
  cat("  - Added FamilySize = SibSp + Parch + 1\n")

  df$IsAlone <- as.integer(df$FamilySize == 1)
  cat("  - Added IsAlone (1 if FamilySize == 1)\n")

  df$Embarked_C <- as.integer(df$Embarked == "C")
  df$Embarked_Q <- as.integer(df$Embarked == "Q")
  cat("  - Encoded Embarked as Embarked_C and Embarked_Q (S is the baseline)\n")

  df$Cabin <- NULL; df$Name <- NULL; df$Ticket <- NULL; df$Embarked <- NULL
  cat("  - Dropped columns: Cabin (mostly missing), Name, Ticket, Embarked\n")
  df
}

train <- prepare(train, "train.csv")

features <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
              "FamilySize", "IsAlone", "Embarked_C", "Embarked_Q")
cat("\nFeatures used in the model:", paste(features, collapse = ", "), "\n")

# ---------- 4. Train logistic regression ----------
# FamilySize is SibSp + Parch + 1, so drop it from the formula to avoid perfect collinearity
model_features <- setdiff(features, "FamilySize")
formula <- as.formula(paste("Survived ~", paste(model_features, collapse = " + ")))
model <- glm(formula, data = train, family = binomial)
cat("\n[4] Trained logistic regression model (glm, family = binomial)\n")
cat("Formula:", deparse(formula, width.cutoff = 500), "\n")
cat("Coefficients:\n")
print(round(coef(model), 4))

# ---------- 5. Training accuracy ----------
train_pred <- as.integer(predict(model, train, type = "response") > 0.5)
cat(sprintf("\n[5] Accuracy on training set: %.4f\n", mean(train_pred == train$Survived)))

# ---------- 6. Predict on test set ----------
test <- load_csv(file.path(data_dir, "test.csv"))
cat(sprintf("\n[6] Loaded test.csv: %d rows, %d columns\n", nrow(test), ncol(test)))
test <- prepare(test, "test.csv")
test$Survived_pred <- as.integer(predict(model, test, type = "response") > 0.5)
cat(sprintf("\nPredicted survivors in test set: %d / %d\n", sum(test$Survived_pred), nrow(test)))

out_path <- file.path(data_dir, "predictions_r.csv")
write.csv(data.frame(PassengerId = test$PassengerId, Survived = test$Survived_pred),
          out_path, row.names = FALSE)
cat("Saved predictions to", out_path, "\n")

# ---------- 7. Test accuracy ----------
# test.csv has no Survived column, so we compare with gender_submission.csv
labels <- load_csv(file.path(data_dir, "gender_submission.csv"))
merged <- merge(test[, c("PassengerId", "Survived_pred")], labels, by = "PassengerId")
cat("\n[7] test.csv has no true labels, so predictions are compared with gender_submission.csv\n")
cat(sprintf("Accuracy on test set (vs gender_submission.csv): %.4f\n",
            mean(merged$Survived_pred == merged$Survived)))

cat("\nDone!\n")