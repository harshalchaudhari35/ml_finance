
# remotes::install_github("zlfccnu/econophysics") #install this using R terminal
library("econophysics")
library("MetNet")
library(doMC)

cc_matrix = read.csv('./data/matrix_shaped_data.csv')

head(cc_matrix)
## set index as the dates
rownames(cc_matrix) <- cc_matrix$close_time
cc_matrix$close_time <- NULL

window_start_dates = array(c('2019-01-01', '2019-06-01', '2019-12-01'))
window_end_dates = array(c('2019-06-01', '2019-12-01', '2020-05-10'))
i = 1

first_window = cc_matrix %>% filter(rownames(cc_matrix) <= window_end_dates[i], rownames(cc_matrix) >= window_start_dates[i])

adjMat = correlation(first_window, type = "pearson", use = "pairwise.complete.obs")
write.csv(adjMat,"./data/adj_mat_from_R.csv")

kuramoto_output = kuramoto(adjMat = adjMat, thread = 4, steps=100)

plot(kuramoto_output[[1]])
