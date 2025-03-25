# analysis.R - 乳腺癌统计分析主程序

# ==== 环境配置 ====
# 安装缺失包（首次运行时需要）
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,    # 数据处理
  here,         # 路径管理
  ggplot2,      # 可视化
  corrplot,     # 相关矩阵
  rmarkdown,    # 报告生成
  knitr,        # 表格格式化
  glue,         # 字符串模板
  fs            # 文件系统操作
)

# ==== 全局配置 ====
config <- list(
  # 路径配置
  data_dir = here("data"),
  raw_data = here("data", "raw", "wdbc.data"),
  processed_data = here("data", "processed", "processed_data.csv"),
  output_dir = here("outputs"),
  report_template = here("scrips", "report_template.Rmd"),
  
  # 分析参数
  seed = 2023,          # 随机种子
  alpha_level = 0.05,   # 显著性水平
  plot_dpi = 300,       # 图形分辨率
  
  # 可视化参数
  color_palette = c(B = "#4E79A7", M = "#E15759")  # 颜色方案
)

# ==== 函数定义 ====

#' 初始化项目环境
initialize_environment <- function(cfg) {
  # 创建输出目录
  dir_create(cfg$output_dir)
  dir_create(path(cfg$output_dir, "visualizations"))
  dir_create(path(cfg$output_dir, "reports"))
  
  # 设置随机种子
  set.seed(cfg$seed)
  
  # 验证文件存在性
  required_files <- c(cfg$processed_data, cfg$report_template)
  file_exists <- map_lgl(required_files, file.exists)
  if (!all(file_exists)) {
    missing_files <- required_files[!file_exists]
    stop(glue("关键文件缺失: {paste(missing_files, collapse=', ')}"))
  }
}

#' 加载预处理数据
load_processed_data <- function(file_path) {
  tryCatch({
    df <- read_csv(
      file_path,
      col_types = cols(
        ID = col_character(),
        Diagnosis = col_factor(levels = c("B", "M")),
        .default = col_double()
      )
    )
    message(glue("成功加载数据: {nrow(df)} 行 × {ncol(df)} 列"))
    return(df)
  }, error = function(e) {
    stop(glue("数据加载失败: {e$message}"))
  })
}

#' 执行诊断结果分析
analyze_diagnosis_distribution <- function(data, cfg) {
  # 统计计算
  stats <- data %>%
    count(Diagnosis, .drop = FALSE) %>%
    mutate(
      Percent = n / sum(n) * 100,
      Percent = round(Percent, 1)
    )
  
  # 生成可视化
  plot <- ggplot(stats, aes(x = Diagnosis, y = n, fill = Diagnosis)) +
    geom_col(width = 0.6, show.legend = FALSE) +
    geom_text(aes(label = glue("{n}\n({Percent}%)")), 
              vjust = -0.3, size = 5, color = "black") +
    scale_fill_manual(values = cfg$color_palette) +
    labs(
      title = "乳腺癌诊断结果分布",
      x = "诊断类别",
      y = "病例数量"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      panel.grid.major.x = element_blank()
    )
  
  # 保存图形
  ggsave(
    path(cfg$output_dir, "visualizations", "diagnosis_distribution.png"),
    plot,
    width = 8,
    height = 6,
    dpi = cfg$plot_dpi
  )
  
  return(list(stats = stats, plot = plot))
}

#' 生成分析报告
generate_analysis_report <- function(data, cfg, analysis_results) {
  render_args <- list(
    input = cfg$report_template,
    output_file = path(cfg$output_dir, "reports", "analysis_report.html"),
    params = list(
      dataset = data,
      diagnosis_stats = analysis_results$diagnosis$stats,
      diagnosis_plot = analysis_results$diagnosis$plot,
      config = cfg
    ),
    envir = new.env(parent = globalenv()),
    quiet = FALSE
  )
  
  tryCatch({
    message("\n正在生成分析报告...")
    exec(rmarkdown::render, !!!render_args)
    message(glue("报告已保存至: {render_args$output_file}"))
  }, error = function(e) {
    stop(glue("报告生成失败: {e$message}"))
  })
}

# ==== 主程序 ====
if (sys.nframe() == 0) {
  tryCatch({
    # 初始化环境
    initialize_environment(config)
    
    # 加载数据
    wdbc_data <- load_processed_data(config$processed_data)
    
    # 执行分析
    analysis_results <- list(
      diagnosis = analyze_diagnosis_distribution(wdbc_data, config)
    )
    
    # 生成报告
    generate_analysis_report(wdbc_data, config, analysis_results)
    
    message("\n==== 分析流程成功完成 ====")
  }, error = function(e) {
    message(glue("\n错误发生: {e$message}"))
    message("详细日志请查看:", path(config$output_dir, "error_log.txt"))
    write_lines(e$message, path(config$output_dir, "error_log.txt"))
    quit(status = 1)
  })
}
