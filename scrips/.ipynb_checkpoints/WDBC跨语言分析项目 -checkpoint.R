# WDBC跨语言分析项目 - R统计分析模块
# 功能：统计分析与可视化
# WDBC 统计分析模块
# 功能：统计检验、可视化、报告生成

library(tidyverse)
library(rmarkdown)
library(here)
library(ggplot2)
library(corrplot)

# 安装必要包
# install.packages(c("here", "fs"))


library(fs)

# 1. 初始化项目结构（只需运行一次）
dir_create(here("data/raw"))     # 原始数据目录
dir_create(here("data/processed")) # 清洗后数据
dir_create(here("outputs/reports")) # 分析报告

# 2. 将你的 processed_data.csv 移动到结构化目录
 file_move("/Users/chen/anaconda_projects/processed_data.csv", 
          here("data/processed/processed_data.csv"))

# 3. 路径配置
data_path <- here("data/processed/processed_data.csv")
report_template <- here("reports", "analysis_template.Rmd")
output_dir <- here("outputs/reports")

# 数据加载
wdbc <- read_csv(data_path) %>%
  mutate(
    Diagnosis = factor(Diagnosis, levels = c("B", "M")),
    malignancy_risk = cut(malignancy_score,
                          breaks = c(-Inf, 0.1, 0.3, Inf),
                          labels = c("Low", "Medium", "High"))
  )

# 描述性统计
cat("=== 基础统计量 ===\n")
print(summary(wdbc))

# 组间差异分析
perform_ttest <- function(var) {
  formula <- as.formula(paste(var, "~ Diagnosis"))
  t.test(formula, data = wdbc)
}

significant_vars <- c("radius_mean", "texture_mean", "area_worst")
ttest_results <- lapply(significant_vars, perform_ttest)
names(ttest_results) <- significant_vars

# 可视化分析
## 密度图
ggplot(wdbc, aes(x = radius_mean, fill = Diagnosis)) +
  geom_density(alpha = 0.6) +
  labs(title = "Radius Mean Distribution by Diagnosis",
       x = "Mean Radius", y = "Density") +
  theme_minimal()

ggsave(here("outputs", "visualizations", "radius_density.png"))

## 相关矩阵
cor_matrix <- wdbc %>%
  select(radius_mean:malignancy_score) %>%
  cor(use = "complete.obs")

png(here("outputs", "visualizations", "corr_matrix.png"))
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", addCoef.col = "black")
dev.off()

# 生成交互式报告
render(
  input = report_template,
  output_format = "html_document",
  output_file = "breast_cancer_analysis.html",
  output_dir = output_dir,
  params = list(
    dataset_path = data_path,
    title = "WDBC 诊断分析报告"
  )
)
