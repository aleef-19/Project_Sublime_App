# Load data
df <- read.table("height_weight.txt", header = TRUE)

# Fit linear model
model <- lm(Weight ~ Height, data = df)

# Output model summary
summary(model)

# Plot data and regression line
plot(Weight ~ Height, data = df, main = "Height vs Weight",
     xlab = "Height", ylab = "Weight", pch = 19, col = "blue")
abline(model, col = "red", lwd = 2)
legend("topleft", legend = c("Data", "Regression Line"),
       col = c("blue", "red"), pch = c(19, NA), lty = c(NA, 1), lwd = c(NA, 2))











