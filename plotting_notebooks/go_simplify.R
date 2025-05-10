library(simplifyEnrichment)
library(org.Hs.eg.db)  # Or org.Mm.eg.db for mouse
library(grid)
good_term <- read.csv("resources/good_terms.csv")
mat = GO_similarity(good_term$go_term)
png("figures/simplifyEnrichment_good.png", width = 2400, height = 1500, res = 300)
good_out <- simplifyGO(mat, column_title="")
# Save the result to a PNG file
dev.off()